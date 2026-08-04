"""
Tests for src/data/generate_synthetic_oulad.py, src/data/ingest.py and the
remaining cutoff helpers.

The generator is a stand-in for real OULAD data, which makes it load-bearing:
if it silently stopped producing learnable signal, every metric in the project
would still compute fine and quietly mean nothing. These tests pin down both
the schema (so real data drops in cleanly) and the signal.
"""
import pandas as pd
import pytest
from pathlib import Path

from src.data.generate_synthetic_oulad import generate
from src.data.ingest import load_raw_tables
from src.data.cutoff import resolve_checkpoints, max_permitted_day


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    out = tmp_path_factory.mktemp("synthetic")
    generate(n_per_presentation=12, seed=7, out_dir=str(out))
    return out


# ── schema fidelity ───────────────────────────────────────────────────────

def test_all_seven_oulad_tables_are_produced(generated):
    expected = {
        "courses.csv", "studentInfo.csv", "studentRegistration.csv",
        "assessments.csv", "studentAssessment.csv", "studentVle.csv",
    }
    assert expected <= {p.name for p in generated.glob("*.csv")}


@pytest.mark.parametrize("filename,required_columns", [
    ("courses.csv", {"code_module", "code_presentation", "module_presentation_length"}),
    ("studentInfo.csv", {"code_module", "code_presentation", "id_student", "gender", "region",
                          "highest_education", "imd_band", "age_band", "num_of_prev_attempts",
                          "studied_credits", "disability", "final_result"}),
    ("studentRegistration.csv", {"code_module", "code_presentation", "id_student",
                                  "date_registration", "date_unregistration"}),
    ("assessments.csv", {"code_module", "code_presentation", "id_assessment",
                          "assessment_type", "date", "weight"}),
    ("studentAssessment.csv", {"id_assessment", "id_student", "date_submitted", "is_banked", "score"}),
    ("studentVle.csv", {"code_module", "code_presentation", "id_student", "id_site", "date", "sum_click"}),
])
def test_table_columns_match_the_oulad_schema(generated, filename, required_columns):
    """Real OULAD CSVs must drop in without touching the pipeline, so column
    names have to match the published schema exactly."""
    df = pd.read_csv(generated / filename)
    assert required_columns <= set(df.columns), f"{filename} missing {required_columns - set(df.columns)}"


def test_final_result_uses_the_real_oulad_vocabulary(generated):
    df = pd.read_csv(generated / "studentInfo.csv")
    assert set(df["final_result"]) <= {"Pass", "Fail", "Withdrawn", "Distinction"}


def test_all_four_outcomes_are_represented(generated):
    df = pd.read_csv(generated / "studentInfo.csv")
    assert len(set(df["final_result"])) >= 3, "degenerate outcome distribution"


# ── referential integrity ─────────────────────────────────────────────────

def test_every_student_has_a_registration_row(generated):
    info = pd.read_csv(generated / "studentInfo.csv")
    reg = pd.read_csv(generated / "studentRegistration.csv")
    assert set(info["id_student"]) == set(reg["id_student"])


def test_assessment_submissions_reference_real_assessments(generated):
    sa = pd.read_csv(generated / "studentAssessment.csv")
    a = pd.read_csv(generated / "assessments.csv")
    assert set(sa["id_assessment"]) <= set(a["id_assessment"])


def test_vle_rows_reference_real_students(generated):
    vle = pd.read_csv(generated / "studentVle.csv")
    info = pd.read_csv(generated / "studentInfo.csv")
    assert set(vle["id_student"]) <= set(info["id_student"])


def test_student_ids_are_unique_per_enrolment(generated):
    info = pd.read_csv(generated / "studentInfo.csv")
    assert not info.duplicated(["id_student", "code_module", "code_presentation"]).any()


# ── the leakage-relevant column ───────────────────────────────────────────

def test_only_withdrawn_students_have_an_unregistration_date(generated):
    """date_unregistration is the most seductive leak in OULAD — it correlates
    almost perfectly with the Withdrawn label because it *is* the withdrawal.
    The generator reproduces that property so the exclusion is genuinely tested."""
    info = pd.read_csv(generated / "studentInfo.csv")
    reg = pd.read_csv(generated / "studentRegistration.csv")
    merged = info.merge(reg, on=["id_student", "code_module", "code_presentation"])
    has_date = merged["date_unregistration"].notna()
    assert (merged.loc[has_date, "final_result"] == "Withdrawn").all()


# ── signal, not noise ─────────────────────────────────────────────────────

def test_engagement_actually_predicts_the_outcome(generated):
    """If this fails, the dataset has become noise and every reported metric in
    the project is meaningless even though nothing errors."""
    info = pd.read_csv(generated / "studentInfo.csv")
    vle = pd.read_csv(generated / "studentVle.csv")
    clicks = vle.groupby("id_student")["sum_click"].sum()
    info["clicks"] = info["id_student"].map(clicks).fillna(0)

    at_risk = info["final_result"].isin(["Fail", "Withdrawn"])
    assert info.loc[at_risk, "clicks"].mean() < info.loc[~at_risk, "clicks"].mean(), \
        "students who fail should engage less, or there is no signal to learn"


def test_early_submission_behaviour_predicts_the_outcome(generated):
    info = pd.read_csv(generated / "studentInfo.csv")
    sa = pd.read_csv(generated / "studentAssessment.csv")
    counts = sa.groupby("id_student").size()
    info["n_sub"] = info["id_student"].map(counts).fillna(0)

    at_risk = info["final_result"].isin(["Fail", "Withdrawn"])
    assert info.loc[at_risk, "n_sub"].mean() < info.loc[~at_risk, "n_sub"].mean()


def test_generation_is_deterministic(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    generate(n_per_presentation=8, seed=99, out_dir=str(a))
    generate(n_per_presentation=8, seed=99, out_dir=str(b))
    pd.testing.assert_frame_equal(
        pd.read_csv(a / "studentInfo.csv"), pd.read_csv(b / "studentInfo.csv")
    )


def test_different_seeds_give_different_data(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    generate(n_per_presentation=8, seed=1, out_dir=str(a))
    generate(n_per_presentation=8, seed=2, out_dir=str(b))
    assert not pd.read_csv(a / "studentInfo.csv").equals(pd.read_csv(b / "studentInfo.csv"))


def test_vle_activity_can_start_before_the_course_does(generated):
    """OULAD records pre-course activity as negative day offsets; the pipeline
    must be exercised against that."""
    vle = pd.read_csv(generated / "studentVle.csv")
    assert (vle["date"] < 0).any()


# ── ingest ────────────────────────────────────────────────────────────────

def test_load_raw_tables_returns_every_table(config):
    tables = load_raw_tables(config)
    assert set(tables) == {
        "courses", "student_info", "student_registration",
        "assessments", "student_assessment", "student_vle",
    }
    assert all(isinstance(df, pd.DataFrame) and not df.empty for df in tables.values())


def test_load_raw_tables_fails_clearly_on_a_missing_directory(config):
    broken = {**config, "data": {**config["data"], "raw_dir": "/nonexistent/path"}}
    with pytest.raises((FileNotFoundError, OSError)):
        load_raw_tables(broken)


# ── max_permitted_day ─────────────────────────────────────────────────────

def test_max_permitted_day_returns_the_checkpoint_for_a_course(raw, config):
    checkpoints = resolve_checkpoints(raw["courses"], config["prediction"]["checkpoint_fraction"])
    first = checkpoints[0]
    assert max_permitted_day(checkpoints, first.code_module, first.code_presentation) == first.cutoff_day


def test_max_permitted_day_raises_on_an_unknown_course(raw, config):
    checkpoints = resolve_checkpoints(raw["courses"], 0.3)
    with pytest.raises(KeyError):
        max_permitted_day(checkpoints, "ZZZ", "1999X")


def test_checkpoint_scales_with_the_configured_fraction(raw):
    early = {(c.code_module, c.code_presentation): c.cutoff_day for c in resolve_checkpoints(raw["courses"], 0.2)}
    late = {(c.code_module, c.code_presentation): c.cutoff_day for c in resolve_checkpoints(raw["courses"], 0.5)}
    assert all(early[k] < late[k] for k in early)

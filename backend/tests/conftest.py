import yaml
import pytest
from pathlib import Path
from src.data.ingest import load_raw_tables


@pytest.fixture(scope="session")
def config():
    return yaml.safe_load(open("config/config.yaml"))


@pytest.fixture(scope="session")
def raw(config):
    """Loads the raw tables, failing with an actionable message if the dataset
    has not been generated yet. Without this, a fresh clone running `make test`
    gets a wall of pandas FileNotFoundError tracebacks that don't say what to
    do about it."""
    raw_dir = Path(config["data"]["raw_dir"])
    expected = raw_dir / config["data"]["tables"]["student_info"]
    if not expected.exists():
        pytest.fail(
            f"\n\nDataset not found at {expected}.\n"
            "The data tests need the raw tables. Generate them first:\n\n"
            "    make data       (or: python -m src.data.generate_synthetic_oulad)\n\n"
            "To use the real OULAD data instead, see DATASET.md.\n",
            pytrace=False,
        )
    return load_raw_tables(config)


@pytest.fixture(scope="session")
def sample_student_payload():
    """A realistic, clearly high-risk student used across API tests."""
    return {
        "student_id": "S-TEST-01",
        "gender": "M",
        "region": "South Region",
        "highest_education": "Lower Than A Level",
        "imd_band": "20-30%",
        "age_band": "0-35",
        "num_of_prev_attempts": 0,
        "studied_credits": 60,
        "disability": "N",
        "date_registration": 5,
        "late_registration": 1,
        "vle_total_clicks": 12,
        "vle_active_days": 3,
        "vle_distinct_sites": 2,
        "vle_click_trend": -0.4,
        "vle_days_since_last_click": 25,
        "n_submitted": 0,
        "avg_early_score": -1,
        "pct_on_time": 0,
        "avg_days_early": 0,
    }

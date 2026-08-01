"""
End-to-end data pipeline: raw tables -> checkpoint-filtered data -> feature
table + labels, for a given checkpoint_fraction. This is the function both
training (train.py) and the "early enough to matter" multi-checkpoint
evaluation call — the checkpoint is always a parameter, never hardcoded.
"""
import pandas as pd
from src.data.cutoff import resolve_checkpoints, apply_checkpoint_cutoff
from src.data.features import assemble_feature_table, label_table, FORBIDDEN_FEATURE_COLUMNS


def build_dataset(raw: dict[str, pd.DataFrame], config: dict, checkpoint_fraction: float | None = None):
    """
    Returns (X, y, feature_columns, meta) where:
      X: DataFrame of model-ready features (one row per student-presentation)
      y: Series of the at_risk binary label
      feature_columns: the numeric/categorical columns to actually feed the model
      meta: id/course columns kept alongside X for bookkeeping (not features)
    """
    fraction = checkpoint_fraction if checkpoint_fraction is not None else config["prediction"]["checkpoint_fraction"]
    checkpoints = resolve_checkpoints(raw["courses"], fraction)

    cutoff_result = apply_checkpoint_cutoff(
        student_vle=raw["student_vle"],
        student_assessment=raw["student_assessment"],
        assessments=raw["assessments"],
        checkpoints=checkpoints,
    )

    feature_table = assemble_feature_table(
        student_info=raw["student_info"],
        registration=raw["student_registration"],
        vle_filtered=cutoff_result["student_vle"],
        sa_filtered=cutoff_result["student_assessment"],
        assessments=raw["assessments"],
        eligible_assessment_ids=cutoff_result["eligible_assessment_ids"],
        checkpoints=checkpoints,
    )

    labels = label_table(raw["student_info"], config["target"]["at_risk_outcomes"])

    full = feature_table.merge(labels[["id_student", "code_module", "code_presentation", "at_risk"]],
                                on=["id_student", "code_module", "code_presentation"], how="inner")

    meta_cols = ["id_student", "code_module", "code_presentation"]
    label_col = "at_risk"
    feature_columns = [c for c in full.columns if c not in meta_cols + [label_col]]

    # defensive check: assert none of the forbidden columns ever ended up in the feature set
    leaked = set(feature_columns) & FORBIDDEN_FEATURE_COLUMNS
    assert not leaked, f"Leakage guard tripped — forbidden columns present in features: {leaked}"

    X = full[meta_cols + feature_columns]
    y = full[label_col]
    return X, y, feature_columns, {"checkpoints": checkpoints, "checkpoint_fraction": fraction}

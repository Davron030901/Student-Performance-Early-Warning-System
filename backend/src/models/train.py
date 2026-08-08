"""
Trains the baseline(s) and main model(s), evaluates on a genuinely held-out
module presentation (not just cross-validation), runs the multi-checkpoint
"early enough to matter" comparison, computes SHAP explainability + a
fairness breakdown, and saves everything needed for the API to serve
predictions.

Run: python -m src.models.train
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    recall_score, precision_score, fbeta_score, roc_auc_score,
    average_precision_score, brier_score_loss, confusion_matrix,
)
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.ingest import load_raw_tables
from src.data.pipeline import build_dataset
from src.data.features import CATEGORICAL_COLUMNS, BASELINE_NUMERIC_FEATURES
from src.models.encoding import encode_features
from src.models.explain import (
    compute_shap_values, save_global_importance_plot, behavioural_reference_medians,
)
from src.models.fairness import fairness_report
from src.models.tracking import track_run

HELD_OUT_PRESENTATION = "2014J"  # most recent presentation -> generalization test


def metrics_from_predictions(y_true, y_pred, y_proba) -> dict:
    return {
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "f2": round(fbeta_score(y_true, y_pred, beta=2, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_true, y_proba), 4),
        "pr_auc": round(average_precision_score(y_true, y_proba), 4),
        "brier": round(brier_score_loss(y_true, y_proba), 4),
        "n": int(len(y_true)),
        "positive_rate": round(float(np.mean(y_true)), 4),
    }


def make_xgb_model(config: dict, scale_pos_weight: float):
    p = config["model"]["xgboost_params"]
    return xgb.XGBClassifier(
        n_estimators=p["n_estimators"], max_depth=p["max_depth"], learning_rate=p["learning_rate"],
        subsample=p["subsample"], colsample_bytree=p["colsample_bytree"],
        scale_pos_weight=scale_pos_weight, random_state=config["model"]["random_state"],
        eval_metric="logloss",
    )


def make_rf_model(config: dict):
    p = config["model"]["random_forest_params"]
    return RandomForestClassifier(
        n_estimators=p["n_estimators"], max_depth=p["max_depth"], class_weight="balanced",
        random_state=config["model"]["random_state"],
    )


def cross_validate_model(model_fn, X_enc: pd.DataFrame, y: pd.Series, groups: pd.Series, n_folds: int, seed: int):
    """Grouped stratified CV (group = student). With real OULAD data, the same
    student can appear across multiple presentations, so grouping prevents a
    student's data leaking between train/validation folds. In this synthetic
    dataset every student is unique to one presentation, so this behaves like
    plain StratifiedKFold — the code is written to be correct for real data."""
    skf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_metrics = []
    for train_idx, val_idx in skf.split(X_enc, y, groups=groups):
        model = model_fn()
        model.fit(X_enc.iloc[train_idx], y.iloc[train_idx])
        proba = model.predict_proba(X_enc.iloc[val_idx])[:, 1]
        pred = (proba >= 0.5).astype(int)
        fold_metrics.append(metrics_from_predictions(y.iloc[val_idx], pred, proba))
    avg = {k: round(float(np.mean([m[k] for m in fold_metrics])), 4) for k in fold_metrics[0] if k != "n"}
    avg["n"] = int(np.sum([m["n"] for m in fold_metrics]))
    return avg


def evaluate_trivial_baseline(X_raw: pd.DataFrame, y: pd.Series) -> dict:
    """Flag anyone with zero submissions by the checkpoint."""
    pred = (X_raw["n_submitted"] == 0).astype(int)
    proba = pred.astype(float)  # binary rule has no real probability
    return metrics_from_predictions(y, pred, proba)


def risk_band(prob: float, config: dict) -> str:
    bands = config["risk_bands"]
    if prob <= bands["low_max"]:
        return "Low"
    if prob <= bands["medium_max"]:
        return "Medium"
    return "High"


def run_checkpoint_evaluation(raw: dict, config: dict, fraction: float) -> dict:
    """Rebuilds the dataset at a given checkpoint fraction, trains the
    primary model on the training presentations, and evaluates recall/etc on
    the held-out presentation. Used for the 'early enough to matter' table."""
    X, y, feature_columns, meta = build_dataset(raw, config, checkpoint_fraction=fraction)
    X_enc, _ = encode_features(X, feature_columns)

    is_held_out = (X["code_presentation"] == HELD_OUT_PRESENTATION).values
    X_train, y_train = X_enc[~is_held_out], y[~is_held_out]
    X_test, y_test = X_enc[is_held_out], y[is_held_out]

    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / max(1, n_pos)

    model = make_xgb_model(config, scale_pos_weight)
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    m = metrics_from_predictions(y_test, pred, proba)
    m["checkpoint_fraction"] = fraction
    return m


def main():
    config = yaml.safe_load(open("config/config.yaml"))
    raw = load_raw_tables(config)

    with track_run("full-training-pipeline") as run:
        _run_pipeline(config, raw, run)


def _run_pipeline(config, raw, run):

    print("=" * 70)
    print("1. Building dataset at the primary checkpoint")
    print("=" * 70)
    primary_fraction = config["prediction"]["checkpoint_fraction"]
    X, y, feature_columns, meta = build_dataset(raw, config, checkpoint_fraction=primary_fraction)
    print(f"Checkpoint: {primary_fraction:.0%} of course length | rows={len(X)} | positive rate={y.mean():.3f}")
    run.log_params(
        checkpoint_fraction=primary_fraction,
        held_out_presentation=HELD_OUT_PRESENTATION,
        primary_model=config["model"]["primary"],
        random_state=config["model"]["random_state"],
        n_folds=config["model"]["n_folds"],
        n_rows=len(X),
        n_features=len(feature_columns),
        at_risk_outcomes=",".join(config["target"]["at_risk_outcomes"]),
        **{f"xgb_{k}": v for k, v in config["model"]["xgboost_params"].items()},
    )
    run.log_metrics(prefix="dataset_", positive_rate=float(y.mean()))

    X_enc, model_feature_columns = encode_features(X, feature_columns)
    is_held_out = (X["code_presentation"] == HELD_OUT_PRESENTATION).values
    X_train, y_train, groups_train = X_enc[~is_held_out], y[~is_held_out], X.loc[~is_held_out, "id_student"]
    X_test, y_test = X_enc[is_held_out], y[is_held_out]
    X_train_raw, X_test_raw = X.loc[~is_held_out], X.loc[is_held_out]
    print(f"Train pool (presentations != {HELD_OUT_PRESENTATION}): {len(X_train)} rows")
    print(f"Held-out generalization test ({HELD_OUT_PRESENTATION}): {len(X_test)} rows")

    print("\n" + "=" * 70)
    print("2. Baselines")
    print("=" * 70)
    trivial = evaluate_trivial_baseline(X_test_raw, y_test)
    print("Trivial rule (zero submissions -> at risk), held-out set:", trivial)
    run.log_metrics(prefix="baseline_trivial_", **trivial)

    lr_pipeline_metrics = cross_validate_model(
        lambda: __import__("sklearn.pipeline", fromlist=["Pipeline"]).Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]),
        X_train[BASELINE_NUMERIC_FEATURES], y_train, groups_train,
        config["model"]["n_folds"], config["model"]["random_state"],
    )
    print("Logistic Regression baseline, CV on train pool:", lr_pipeline_metrics)
    run.log_metrics(prefix="baseline_logreg_cv_", **lr_pipeline_metrics)

    print("\n" + "=" * 70)
    print("3. Main models — cross-validated on the training pool")
    print("=" * 70)
    n_pos, n_neg = y_train.sum(), len(y_train) - y_train.sum()
    scale_pos_weight = n_neg / max(1, n_pos)

    xgb_cv = cross_validate_model(lambda: make_xgb_model(config, scale_pos_weight),
                                   X_train, y_train, groups_train,
                                   config["model"]["n_folds"], config["model"]["random_state"])
    print("XGBoost, CV on train pool:", xgb_cv)
    run.log_metrics(prefix="xgboost_cv_", **xgb_cv)

    rf_cv = cross_validate_model(lambda: make_rf_model(config), X_train, y_train, groups_train,
                                  config["model"]["n_folds"], config["model"]["random_state"])
    print("Random Forest, CV on train pool:", rf_cv)
    run.log_metrics(prefix="random_forest_cv_", **rf_cv)

    print("\n" + "=" * 70)
    print("4. Selected model: XGBoost — evaluated on the HELD-OUT presentation")
    print("=" * 70)
    final_model = make_xgb_model(config, scale_pos_weight)
    final_model.fit(X_train, y_train)
    test_proba = final_model.predict_proba(X_test)[:, 1]
    test_pred = (test_proba >= 0.5).astype(int)
    held_out_metrics = metrics_from_predictions(y_test, test_pred, test_proba)
    print("Held-out metrics:", held_out_metrics)
    run.log_metrics(prefix="held_out_", **held_out_metrics)
    cm = confusion_matrix(y_test, test_pred).tolist()
    print("Confusion matrix [[TN,FP],[FN,TP]]:", cm)

    Path(config["paths"]["reports_dir"]).mkdir(exist_ok=True)

    # calibration plot
    from sklearn.calibration import calibration_curve
    frac_pos, mean_pred = calibration_curve(y_test, test_proba, n_bins=10, strategy="quantile")
    plt.figure()
    plt.plot(mean_pred, frac_pos, marker="o", label="Model")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly calibrated")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction actually at-risk")
    plt.title("Calibration — held-out presentation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(config["paths"]["reports_dir"]) / "calibration.png", dpi=120)
    plt.close()

    print("\n" + "=" * 70)
    print("5. 'Early enough to matter' — recall at different checkpoints")
    print("=" * 70)
    checkpoint_results = []
    for frac in config["prediction"]["eval_checkpoint_fractions"]:
        r = run_checkpoint_evaluation(raw, config, frac)
        checkpoint_results.append(r)
        print(f"checkpoint={frac:.0%} -> recall={r['recall']} precision={r['precision']} "
              f"f2={r['f2']} roc_auc={r['roc_auc']}")
        run.log_metrics(prefix=f"checkpoint_{int(frac*100)}pct_",
                        recall=r["recall"], precision=r["precision"],
                        f2=r["f2"], roc_auc=r["roc_auc"])
    pd.DataFrame(checkpoint_results).to_csv(
        Path(config["paths"]["reports_dir"]) / "checkpoint_comparison.csv", index=False
    )

    print("\n" + "=" * 70)
    print("6. Explainability (SHAP) on the held-out set")
    print("=" * 70)
    shap_values, explainer = compute_shap_values(final_model, X_test)
    save_global_importance_plot(shap_values, X_test, str(Path(config["paths"]["reports_dir"]) / "shap_global_importance.png"))
    print("Saved global SHAP importance plot to reports/shap_global_importance.png")

    print("\n" + "=" * 70)
    print("7. Fairness check on the held-out set")
    print("=" * 70)
    fairness_df = fairness_report(y_test, pd.Series(test_pred, index=y_test.index),
                                   X_test_raw, config["fairness"]["sensitive_columns"])
    fairness_df.to_csv(Path(config["paths"]["reports_dir"]) / "fairness_report.csv", index=False)
    print(fairness_df.to_string(index=False))

    print("\n" + "=" * 70)
    print("8. Refitting final deployed model on ALL available data")
    print("=" * 70)
    # honest metrics already captured above on the held-out split; now use all
    # the data we have for the artifact that will actually serve predictions
    final_deployed_model = make_xgb_model(config, scale_pos_weight)
    final_deployed_model.fit(X_enc, y)

    model_path = Path(config["paths"]["model_artifact"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_deployed_model, model_path)

    metadata = {
        "model_version": config["api"]["model_version"],
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "checkpoint_fraction": primary_fraction,
        "feature_columns": feature_columns,
        "model_feature_columns": model_feature_columns,
        "categorical_columns": CATEGORICAL_COLUMNS,
        "risk_bands": config["risk_bands"],
        # medians let inference say "below average" with something to compare against
        "reference_medians": behavioural_reference_medians(X_enc, feature_columns),
        "held_out_metrics": held_out_metrics,
        "cv_metrics": {"logistic_regression_baseline": lr_pipeline_metrics, "xgboost": xgb_cv, "random_forest": rf_cv},
        "trivial_baseline_metrics": trivial,
        "confusion_matrix_held_out": cm,
        "n_training_rows": int(len(X)),
    }
    with open(config["paths"]["metadata_artifact"], "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    for artifact in [model_path, config["paths"]["metadata_artifact"],
                     Path(config["paths"]["reports_dir"]) / "checkpoint_comparison.csv",
                     Path(config["paths"]["reports_dir"]) / "fairness_report.csv",
                     Path(config["paths"]["reports_dir"]) / "calibration.png",
                     Path(config["paths"]["reports_dir"]) / "shap_global_importance.png"]:
        run.log_artifact(artifact)

    print(f"\nSaved model artifact -> {model_path}")
    print(f"Saved metadata -> {config['paths']['metadata_artifact']}")
    print("\nDone.")


if __name__ == "__main__":
    main()

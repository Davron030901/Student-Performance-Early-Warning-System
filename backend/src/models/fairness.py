"""
Fairness check: recall/precision of the at-risk classifier, broken out by
each sensitive column configured in config.yaml. This does NOT apply any
mitigation — per the brief, the requirement is to document disparities, not
necessarily solve them in this prototype.
"""
import pandas as pd
from sklearn.metrics import recall_score, precision_score


def fairness_report(y_true: pd.Series, y_pred: pd.Series, demographics: pd.DataFrame,
                     sensitive_columns: list[str]) -> pd.DataFrame:
    rows = []
    df = demographics.copy()
    df["y_true"] = y_true.values
    df["y_pred"] = y_pred.values

    for col in sensitive_columns:
        if col not in df.columns:
            continue
        for group_value, g in df.groupby(col):
            if len(g) < 10:
                continue  # too few samples for a meaningful group-level metric
            rows.append({
                "sensitive_attribute": col,
                "group": group_value,
                "n": len(g),
                "base_rate_at_risk": g["y_true"].mean(),
                "recall": recall_score(g["y_true"], g["y_pred"], zero_division=0),
                "precision": precision_score(g["y_true"], g["y_pred"], zero_division=0),
            })
    return pd.DataFrame(rows).sort_values(["sensitive_attribute", "group"])

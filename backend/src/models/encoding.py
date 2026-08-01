"""
Shared feature-encoding utilities. train.py uses encode_features() on the
full training table; the API uses align_to_model_columns() on a single
incoming student, reindexed against the exact column list the model was
trained on (saved in metadata.json). Keeping both in one module guarantees
they can never silently drift apart.
"""
import pandas as pd
from src.data.features import CATEGORICAL_COLUMNS


def sanitize_column_name(name: str) -> str:
    """XGBoost forbids '[', ']', '<' in feature names (OULAD's age_band
    category '55<=' triggers this)."""
    return str(name).replace("<", "lt").replace(">", "gt").replace("[", "(").replace("]", ")")


def encode_features(X: pd.DataFrame, feature_columns: list[str]):
    encoded = pd.get_dummies(X[feature_columns], columns=CATEGORICAL_COLUMNS, dummy_na=False)
    encoded.columns = [sanitize_column_name(c) for c in encoded.columns]
    return encoded, list(encoded.columns)


def align_to_model_columns(row: pd.DataFrame, model_feature_columns: list[str]) -> pd.DataFrame:
    """One-hot encodes a single (or small batch of) incoming student row(s)
    the same way training data was encoded, then reindexes to the exact
    column set the model expects — unseen categories simply contribute all
    zeros for their column group, matching handle_unknown='ignore' semantics."""
    encoded = pd.get_dummies(row, columns=CATEGORICAL_COLUMNS, dummy_na=False)
    encoded.columns = [sanitize_column_name(c) for c in encoded.columns]
    return encoded.reindex(columns=model_feature_columns, fill_value=0)

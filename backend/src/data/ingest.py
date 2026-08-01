"""Loads the raw OULAD-schema CSV tables listed in config.yaml."""
from pathlib import Path
import pandas as pd


def load_raw_tables(config: dict) -> dict[str, pd.DataFrame]:
    raw_dir = Path(config["data"]["raw_dir"])
    tables = config["data"]["tables"]
    return {
        "courses": pd.read_csv(raw_dir / tables["courses"]),
        "student_info": pd.read_csv(raw_dir / tables["student_info"]),
        "student_registration": pd.read_csv(raw_dir / tables["student_registration"]),
        "assessments": pd.read_csv(raw_dir / tables["assessments"]),
        "student_assessment": pd.read_csv(raw_dir / tables["student_assessment"]),
        "student_vle": pd.read_csv(raw_dir / tables["student_vle"]),
    }

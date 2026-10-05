"""
Data Validation & Security Module
Provides schema validation, integrity hashing (SHA-256), and security sanity checks.
"""

import hashlib
import os
import pandas as pd
from typing import Dict, Any, List, Tuple


REQUIRED_RAW_COLUMNS = [
    "Student_ID", "Age", "Gender", "Department", "Year", "Semester",
    "Parental_Education", "Internet_Access", "Study_Hours_Week",
    "LMS_Hours_Week", "Attendance_Pct", "Assignment_Score",
    "Mid_Grade", "Final_Grade", "Pass_Fail", "At_Risk_Flag", "Weekly_Performance"
]

REQUIRED_PROCESSED_COLUMNS = REQUIRED_RAW_COLUMNS + [
    "Assignment_Score_Norm", "Mid_Grade_Norm", "Final_Grade_Norm",
    "LMS_Category", "Attendance_Category", "Grade_Band"
]

FEATURE_COLUMNS = [
    "Study_Hours_Week", "LMS_Hours_Week", "Attendance_Pct",
    "Assignment_Score", "Mid_Grade", "Year"
]

TARGET_COLUMN = "Pass_Fail"


def calculate_file_sha256(filepath: str) -> str:
    """Calculates SHA-256 cryptographic hash of a file for integrity verification."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def validate_dataframe_schema(df: pd.DataFrame, is_processed: bool = True) -> Tuple[bool, List[str]]:
    """
    Validates DataFrame schema, column existence, range checks, and null limits.
    Returns (is_valid, list_of_errors).
    """
    errors = []
    required_cols = REQUIRED_PROCESSED_COLUMNS if is_processed else REQUIRED_RAW_COLUMNS

    # 1. Check missing columns
    for col in required_cols:
        if col not in df.columns:
            errors.append(f"Missing required column: {col}")

    if errors:
        return False, errors

    # 2. Check value range boundaries
    if "Age" in df.columns:
        if (df["Age"] < 15).any() or (df["Age"] > 80).any():
            errors.append("Age column contains values out of range (15-80).")

    if "Attendance_Pct" in df.columns and is_processed:
        if (df["Attendance_Pct"] < 0).any() or (df["Attendance_Pct"] > 100).any():
            errors.append("Attendance_Pct contains values outside [0, 100].")

    if "Final_Grade" in df.columns and is_processed:
        if (df["Final_Grade"] < 0).any() or (df["Final_Grade"] > 100).any():
            errors.append("Final_Grade contains values outside [0, 100].")

    if "Pass_Fail" in df.columns:
        invalid_pf = df["Pass_Fail"].dropna().isin([0, 1])
        if not invalid_pf.all():
            errors.append("Pass_Fail column must contain only 0 or 1.")

    # 3. For processed data, ensure zero missing values in feature columns
    if is_processed:
        missing_counts = df[FEATURE_COLUMNS].isnull().sum()
        cols_with_nulls = missing_counts[missing_counts > 0]
        if not cols_with_nulls.empty:
            errors.append(f"Processed features contain unexpected nulls: {cols_with_nulls.to_dict()}")

    return len(errors) == 0, errors


def validate_single_input(payload: Dict[str, Any]) -> Tuple[bool, str]:
    """Validates a single prediction input dictionary for security and type safety."""
    for feature in FEATURE_COLUMNS:
        if feature not in payload:
            return False, f"Missing required feature: {feature}"

        val = payload[feature]
        if not isinstance(val, (int, float)):
            return False, f"Feature '{feature}' must be numeric, got {type(val).__name__}"

    # Value range checks
    if not (0 <= payload["Study_Hours_Week"] <= 100):
        return False, "Study_Hours_Week must be between 0 and 100"
    if not (0 <= payload["LMS_Hours_Week"] <= 100):
        return False, "LMS_Hours_Week must be between 0 and 100"
    if not (0 <= payload["Attendance_Pct"] <= 100):
        return False, "Attendance_Pct must be between 0 and 100"
    if not (0 <= payload["Assignment_Score"] <= 100):
        return False, "Assignment_Score must be between 0 and 100"
    if not (0 <= payload["Mid_Grade"] <= 100):
        return False, "Mid_Grade must be between 0 and 100"
    if payload["Year"] not in [1, 2, 3, 4, 5]:
        return False, "Year must be an integer between 1 and 5"

    return True, "Valid"


if __name__ == "__main__":
    test_path = "data/processed/student_performance_clean.csv"
    if os.path.exists(test_path):
        df_test = pd.read_csv(test_path)
        valid, errs = validate_dataframe_schema(df_test, is_processed=True)
        sha = calculate_file_sha256(test_path)
        print(f"Data validation status: {'PASSED' if valid else 'FAILED'}")
        print(f"Dataset SHA-256: {sha}")
        if errs:
            print("Errors:", errs)
    else:
        print("Processed dataset not found for validation test.")

"""
Unit & Data Quality Tests for Preprocessing and Validation
"""

import os
import pytest
import pandas as pd
import numpy as np

from src.data_preprocessing import generate_raw_dataset, clean_and_preprocess_data
from src.data_validation import (
    validate_dataframe_schema,
    calculate_file_sha256,
    REQUIRED_RAW_COLUMNS,
    REQUIRED_PROCESSED_COLUMNS,
    FEATURE_COLUMNS
)


def test_raw_data_generation(tmp_path):
    """Test synthetic raw data generation produces expected shape and columns."""
    raw_file = str(tmp_path / "raw_test.csv")
    df = generate_raw_dataset(n_samples=100, seed=42, output_path=raw_file)

    assert os.path.exists(raw_file)
    assert len(df) >= 100
    for col in REQUIRED_RAW_COLUMNS:
        assert col in df.columns


def test_data_cleaning_and_feature_engineering(tmp_path):
    """Test data cleaning removes duplicates, handles missing values, and creates engineered features."""
    raw_file = str(tmp_path / "raw_test.csv")
    clean_file = str(tmp_path / "clean_test.csv")

    generate_raw_dataset(n_samples=200, seed=42, output_path=raw_file)
    df_clean = clean_and_preprocess_data(input_path=raw_file, output_path=clean_file)

    assert os.path.exists(clean_file)
    # Check duplicate removal
    assert df_clean.duplicated().sum() == 0

    # Check zero nulls in feature columns
    assert df_clean[FEATURE_COLUMNS].isnull().sum().sum() == 0

    # Check engineered features
    assert "LMS_Category" in df_clean.columns
    assert "Attendance_Category" in df_clean.columns
    assert "Grade_Band" in df_clean.columns
    assert "At_Risk_Flag" in df_clean.columns

    # Check normalization range
    assert df_clean["Assignment_Score_Norm"].min() >= 0
    assert df_clean["Assignment_Score_Norm"].max() <= 100


def test_schema_validation():
    """Test DataFrame schema validator catches missing columns and invalid boundaries."""
    valid_data = pd.DataFrame({
        "Student_ID": ["STU1001"],
        "Age": [20],
        "Gender": ["Male"],
        "Department": ["Computer Science"],
        "Year": [1],
        "Semester": ["Sem-1"],
        "Parental_Education": ["Graduate"],
        "Internet_Access": ["Yes"],
        "Study_Hours_Week": [10.0],
        "LMS_Hours_Week": [5.0],
        "Attendance_Pct": [85.0],
        "Assignment_Score": [75.0],
        "Mid_Grade": [70.0],
        "Final_Grade": [72.0],
        "Pass_Fail": [1],
        "At_Risk_Flag": [0],
        "Weekly_Performance": ["50;60;70"],
        "Assignment_Score_Norm": [75.0],
        "Mid_Grade_Norm": [70.0],
        "Final_Grade_Norm": [72.0],
        "LMS_Category": ["Medium (3-7 hrs)"],
        "Attendance_Category": ["High (>85%)"],
        "Grade_Band": ["Good (B)"]
    })

    is_valid, errors = validate_dataframe_schema(valid_data, is_processed=True)
    assert is_valid is True
    assert len(errors) == 0

    # Test corrupted data
    invalid_data = valid_data.copy()
    invalid_data.loc[0, "Attendance_Pct"] = 150.0  # Out of range
    is_valid, errors = validate_dataframe_schema(invalid_data, is_processed=True)
    assert is_valid is False
    assert any("Attendance_Pct" in e for e in errors)


def test_file_sha256_hash(tmp_path):
    """Test SHA-256 hash calculation and sensitivity to file changes."""
    test_file = tmp_path / "hash_test.txt"
    test_file.write_text("student data content")

    hash1 = calculate_file_sha256(str(test_file))
    assert len(hash1) == 64

    # Tampering test
    test_file.write_text("student data content modified")
    hash2 = calculate_file_sha256(str(test_file))
    assert hash1 != hash2

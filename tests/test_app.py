"""
Inference and Application Logic Tests
"""

import pytest
import pandas as pd
from src.predict import predict_student_risk, batch_predict, categorize_risk
from src.data_validation import validate_single_input


def test_categorize_risk():
    """Test risk category binning."""
    assert categorize_risk(0.90) == "Critical Risk"
    assert categorize_risk(0.60) == "High Risk"
    assert categorize_risk(0.35) == "Moderate Risk"
    assert categorize_risk(0.10) == "Low Risk"


def test_single_prediction_valid():
    """Test valid single student risk inference."""
    payload = {
        "Study_Hours_Week": 12.0,
        "LMS_Hours_Week": 8.0,
        "Attendance_Pct": 90.0,
        "Assignment_Score": 85.0,
        "Mid_Grade": 80.0,
        "Year": 2
    }
    result = predict_student_risk(payload, model_path="models/model.pkl")

    assert result["status"] == "success"
    assert result["predicted_pass"] is True
    assert result["prediction_label"] == "Pass"
    assert 0.0 <= result["risk_score"] <= 1.0
    assert result["risk_category"] in ["Low Risk", "Moderate Risk", "High Risk", "Critical Risk"]


def test_single_prediction_validation_rejection():
    """Test malformed and out-of-bound inputs are safely rejected."""
    # Out of bounds attendance (>100)
    bad_payload = {
        "Study_Hours_Week": 10.0,
        "LMS_Hours_Week": 5.0,
        "Attendance_Pct": 150.0,  # Invalid
        "Assignment_Score": 80.0,
        "Mid_Grade": 75.0,
        "Year": 2
    }
    with pytest.raises(ValueError, match="Input validation error"):
        predict_student_risk(bad_payload, model_path="models/model.pkl")

    # Missing feature
    missing_payload = {
        "Study_Hours_Week": 10.0,
        "LMS_Hours_Week": 5.0,
        "Attendance_Pct": 80.0
    }
    with pytest.raises(ValueError, match="Input validation error"):
        predict_student_risk(missing_payload, model_path="models/model.pkl")


def test_batch_prediction():
    """Test batch prediction function on DataFrame."""
    sample_df = pd.DataFrame([
        {
            "Student_ID": "STU1001",
            "Study_Hours_Week": 15.0,
            "LMS_Hours_Week": 10.0,
            "Attendance_Pct": 95.0,
            "Assignment_Score": 90.0,
            "Mid_Grade": 88.0,
            "Year": 3
        },
        {
            "Student_ID": "STU1002",
            "Study_Hours_Week": 2.0,
            "LMS_Hours_Week": 1.0,
            "Attendance_Pct": 45.0,
            "Assignment_Score": 30.0,
            "Mid_Grade": 35.0,
            "Year": 1
        }
    ])

    res = batch_predict(sample_df, model_path="models/model.pkl")
    assert "Risk_Score" in res.columns
    assert "Risk_Category" in res.columns
    assert "Predicted_Pass" in res.columns
    assert len(res) == 2
    # The struggling student should have higher risk score than the top performer
    assert res.loc[1, "Risk_Score"] > res.loc[0, "Risk_Score"]

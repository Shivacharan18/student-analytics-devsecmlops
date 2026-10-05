"""
Prediction & Inference Module
Provides single-sample and batch prediction functions with input validation.
"""

import os
import sys
import pandas as pd
from typing import Dict, Any, Union, List

# Add parent directory to path to allow direct execution
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_validation import validate_single_input, FEATURE_COLUMNS
from src.model import load_model_artifact

# Global cache for loaded model
_MODEL_CACHE = None


def get_model(model_path: str = "models/model.pkl", verify_integrity: bool = True):
    """Loads and caches the model artifact."""
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        _MODEL_CACHE, _ = load_model_artifact(model_path=model_path, verify_integrity=verify_integrity)
    return _MODEL_CACHE


def categorize_risk(risk_score: float) -> str:
    """Categorizes risk score (prob of fail) into standard risk tiers."""
    if risk_score >= 0.75:
        return "Critical Risk"
    elif risk_score >= 0.50:
        return "High Risk"
    elif risk_score >= 0.25:
        return "Moderate Risk"
    else:
        return "Low Risk"


def predict_student_risk(student_data: Dict[str, Any], model_path: str = "models/model.pkl") -> Dict[str, Any]:
    """
    Validates student input features and computes pass/fail prediction and failure risk score.
    """
    is_valid, msg = validate_single_input(student_data)
    if not is_valid:
        raise ValueError(f"Input validation error: {msg}")

    model = get_model(model_path)
    df_single = pd.DataFrame([student_data])[FEATURE_COLUMNS]

    # Predict prob of fail (class 0)
    prob_pass = float(model.predict_proba(df_single)[0, 1])
    prob_fail = float(1.0 - prob_pass)
    predicted_pass = int(model.predict(df_single)[0])

    return {
        "predicted_pass": bool(predicted_pass),
        "prediction_label": "Pass" if predicted_pass == 1 else "Fail",
        "pass_probability": round(prob_pass, 4),
        "risk_score": round(prob_fail, 4),
        "risk_category": categorize_risk(prob_fail),
        "status": "success"
    }


def batch_predict(df: pd.DataFrame, model_path: str = "models/model.pkl") -> pd.DataFrame:
    """
    Computes batch predictions and adds risk scores and categories to the DataFrame.
    """
    model = get_model(model_path)
    for feat in FEATURE_COLUMNS:
        if feat not in df.columns:
            raise ValueError(f"Missing feature in batch DataFrame: {feat}")

    X = df[FEATURE_COLUMNS]
    probs_pass = model.predict_proba(X)[:, 1]
    probs_fail = 1.0 - probs_pass
    preds = model.predict(X)

    res_df = df.copy()
    res_df["Predicted_Pass"] = preds
    res_df["Risk_Score"] = probs_fail.round(4)
    res_df["Risk_Category"] = [categorize_risk(r) for r in probs_fail]

    return res_df


if __name__ == "__main__":
    sample = {
        "Study_Hours_Week": 4.5,
        "LMS_Hours_Week": 2.0,
        "Attendance_Pct": 62.0,
        "Assignment_Score": 48.0,
        "Mid_Grade": 45.0,
        "Year": 1
    }
    print("Testing single inference with sample data:")
    try:
        res = predict_student_risk(sample)
        print(res)
    except Exception as e:
        print(f"Error during prediction test: {e}")

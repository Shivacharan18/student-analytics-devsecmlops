"""
ML Model Definition & Serialization Module
Defines Scikit-Learn pipelines, model artifact persistence, and integrity verification.
"""

import hashlib
import json
import os
import joblib
from datetime import datetime, timezone
from typing import Any, Dict, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


def build_pipeline(algorithm: str = "LogisticRegression", **hyperparams) -> Pipeline:
    """Builds a scikit-learn preprocessing and estimator pipeline."""
    if algorithm == "LogisticRegression":
        clf_params = {"max_iter": 1000, "random_state": 42, "C": 1.0}
        clf_params.update(hyperparams)
        clf = LogisticRegression(**clf_params)
    elif algorithm == "DecisionTreeClassifier":
        clf_params = {"max_depth": 5, "random_state": 42, "min_samples_leaf": 10}
        clf_params.update(hyperparams)
        clf = DecisionTreeClassifier(**clf_params)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", clf)
    ])
    return pipeline


def save_model_artifact(model: Any, model_path: str = "models/model.pkl", metadata: Dict[str, Any] = None) -> str:
    """Saves model artifact along with SHA-256 integrity hash and metadata JSON."""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)

    # Compute SHA-256
    sha256_hash = hashlib.sha256()
    with open(model_path, "rb") as f:
        for block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(block)
    checksum = sha256_hash.hexdigest()

    meta = {
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "sha256": checksum,
        "features": ["Study_Hours_Week", "LMS_Hours_Week", "Attendance_Pct", "Assignment_Score", "Mid_Grade", "Year"],
        "target": "Pass_Fail",
        "custom_metadata": metadata or {}
    }

    meta_path = f"{os.path.splitext(model_path)[0]}_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return checksum


def load_model_artifact(model_path: str = "models/model.pkl", verify_integrity: bool = True) -> Tuple[Any, Dict[str, Any]]:
    """Loads model artifact and optionally verifies cryptographic integrity against metadata."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model artifact not found at {model_path}")

    meta_path = f"{os.path.splitext(model_path)[0]}_metadata.json"
    metadata = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    if verify_integrity and metadata.get("sha256"):
        sha256_hash = hashlib.sha256()
        with open(model_path, "rb") as f:
            for block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(block)
        current_hash = sha256_hash.hexdigest()
        if current_hash != metadata["sha256"]:
            raise ValueError(f"Model integrity verification failed! Expected {metadata['sha256']}, got {current_hash}")

    model = joblib.load(model_path)
    return model, metadata

"""
ML Model Training & Quality Gate Tests
"""

import os
import json
import pytest
from src.model import build_pipeline, save_model_artifact, load_model_artifact
from src.train import train_model


def test_build_pipeline():
    """Test pipeline initialization with valid and invalid algorithms."""
    lr_pipe = build_pipeline("LogisticRegression")
    assert "scaler" in lr_pipe.named_steps
    assert "classifier" in lr_pipe.named_steps

    dt_pipe = build_pipeline("DecisionTreeClassifier")
    assert "scaler" in dt_pipe.named_steps
    assert "classifier" in dt_pipe.named_steps

    with pytest.raises(ValueError):
        build_pipeline("NonExistentAlgorithm")


def test_model_training_and_gates(tmp_path):
    """Test complete training run and quality gate verification."""
    model_path = str(tmp_path / "test_model.pkl")
    data_path = "data/processed/student_performance_clean.csv"

    if os.path.exists(data_path):
        metrics = train_model(
            data_path=data_path,
            model_save_path=model_path,
            algorithm="LogisticRegression",
            enforce_quality_gates=True
        )

        assert os.path.exists(model_path)
        assert metrics["accuracy"] >= 0.80
        assert metrics["recall"] >= 0.85
        assert metrics["roc_auc"] >= 0.85

        # Verify metadata
        meta_path = str(tmp_path / "test_model_metadata.json")
        assert os.path.exists(meta_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
            assert "sha256" in meta
            assert meta["target"] == "Pass_Fail"


def test_model_tamper_detection(tmp_path):
    """Test that load_model_artifact detects tampered model files."""
    model_path = str(tmp_path / "tamper_test.pkl")
    pipe = build_pipeline("LogisticRegression")
    save_model_artifact(pipe, model_path=model_path)

    # Tamper with the model file
    with open(model_path, "ab") as f:
        f.write(b"corrupt_byte_injection")

    with pytest.raises(ValueError, match="integrity verification failed"):
        load_model_artifact(model_path=model_path, verify_integrity=True)

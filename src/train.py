"""
Model Training Module
Executes model training, evaluation, MLOps quality gate checks, and artifact saving.
"""

import os
import sys
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score

# Add parent directory to path to allow direct script execution
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_validation import validate_dataframe_schema, FEATURE_COLUMNS, TARGET_COLUMN
from src.model import build_pipeline, save_model_artifact


def train_model(data_path: str = "data/processed/student_performance_clean.csv",
                model_save_path: str = "models/model.pkl",
                algorithm: str = "LogisticRegression",
                enforce_quality_gates: bool = True) -> dict:
    """Trains the model, evaluates metrics, validates quality gates, and persists the artifact."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file {data_path} not found.")

    df = pd.read_csv(data_path)

    # 1. Validate data schema
    is_valid, errors = validate_dataframe_schema(df, is_processed=True)
    if not is_valid:
        raise ValueError(f"Data validation failed prior to training: {errors}")

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    # 2. Stratified train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # 3. Build & fit pipeline
    pipeline = build_pipeline(algorithm=algorithm)
    pipeline.fit(X_train, y_train)

    # 4. Predictions & Evaluation
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred))
    rec = float(recall_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred))
    auc = float(roc_auc_score(y_test, y_prob))
    cv_mean = float(cross_val_score(pipeline, X_train, y_train, cv=5, scoring="accuracy").mean())

    metrics = {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(auc, 4),
        "cv_accuracy_mean": round(cv_mean, 4),
        "test_samples": len(y_test)
    }

    print("\n" + "=" * 50)
    print(f"MODEL TRAINING COMPLETED — {algorithm}")
    print("=" * 50)
    for k, v in metrics.items():
        print(f"  {k:20s}: {v}")
    print("=" * 50)

    # 5. MLOps Quality Gates
    if enforce_quality_gates:
        min_acc = 0.80
        min_rec = 0.85
        min_auc = 0.85

        failures = []
        if acc < min_acc:
            failures.append(f"Accuracy {acc:.4f} < {min_acc}")
        if rec < min_rec:
            failures.append(f"Recall {rec:.4f} < {min_rec}")
        if auc < min_auc:
            failures.append(f"ROC-AUC {auc:.4f} < {min_auc}")

        if failures:
            raise RuntimeError(f"Model failed MLOps Quality Gate: {failures}")
        print("[OK] MLOps Quality Gates Passed!")

    # 6. Save Artifact
    checksum = save_model_artifact(pipeline, model_save_path, metadata=metrics)
    print(f"[OK] Model artifact saved to: {model_save_path}")
    print(f"     SHA-256 Checksum: {checksum}\n")

    return metrics


if __name__ == "__main__":
    train_model()

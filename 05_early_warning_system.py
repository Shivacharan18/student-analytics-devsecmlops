"""
=============================================================
SCRIPT 05 — Early Warning System (ML Model)
Case Study: Interactive Data Analytics and Visualization
           of Student Academic Performance
=============================================================
Tasks covered:
  • Logistic Regression (primary model)
  • Decision Tree Classifier (comparison)
  • Train/Test split (80/20)
  • Confusion Matrix
  • Precision, Recall, F1-Score, ROC-AUC
  • Feature Importance Chart
  • At-Risk student report CSV
  • All outputs saved to outputs/model/
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
import warnings, os
warnings.filterwarnings("ignore")
os.makedirs("outputs/model", exist_ok=True)

plt.rcParams.update({"font.family": "DejaVu Sans",
                     "axes.spines.top": False,
                     "axes.spines.right": False,
                     "figure.dpi": 120})

# ── Load Data ─────────────────────────────────────────────────────────────────
df = pd.read_csv("data/student_performance_clean.csv")

# ── Feature Engineering ───────────────────────────────────────────────────────
features = ["Study_Hours_Week", "LMS_Hours_Week", "Attendance_Pct",
            "Assignment_Score", "Mid_Grade", "Year"]
target   = "Pass_Fail"

X = df[features].copy()
y = df[target].copy()

print("=" * 65)
print("EARLY WARNING SYSTEM — Model Training")
print("=" * 65)
print(f"Dataset shape  : {X.shape}")
print(f"Pass (1): {y.sum()}  |  Fail (0): {(y==0).sum()}")
print(f"Class balance  : {y.mean()*100:.1f}% Pass / {(1-y.mean())*100:.1f}% Fail\n")

# ── Train/Test Split ──────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# ── Scale features ────────────────────────────────────────────────────────────
scaler  = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# ─────────────────────────────────────────────────────────────────────────────
# MODEL 1: Logistic Regression
# ─────────────────────────────────────────────────────────────────────────────
lr = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
lr.fit(X_train_s, y_train)
y_pred_lr     = lr.predict(X_test_s)
y_prob_lr     = lr.predict_proba(X_test_s)[:, 1]

acc_lr  = accuracy_score(y_test, y_pred_lr)
prec_lr = precision_score(y_test, y_pred_lr)
rec_lr  = recall_score(y_test, y_pred_lr)
f1_lr   = f1_score(y_test, y_pred_lr)
auc_lr  = roc_auc_score(y_test, y_prob_lr)
cv_lr   = cross_val_score(lr, X_train_s, y_train, cv=5, scoring="accuracy").mean()

print("LOGISTIC REGRESSION")
print("-" * 40)
print(f"  Accuracy       : {acc_lr*100:.2f}%")
print(f"  Precision      : {prec_lr*100:.2f}%")
print(f"  Recall         : {rec_lr*100:.2f}%")
print(f"  F1-Score       : {f1_lr*100:.2f}%")
print(f"  ROC-AUC        : {auc_lr:.4f}")
print(f"  5-Fold CV Acc  : {cv_lr*100:.2f}%")
print(f"\n{classification_report(y_test, y_pred_lr, target_names=['Fail','Pass'])}")

# ─────────────────────────────────────────────────────────────────────────────
# MODEL 2: Decision Tree
# ─────────────────────────────────────────────────────────────────────────────
dt = DecisionTreeClassifier(max_depth=5, random_state=42, min_samples_leaf=10)
dt.fit(X_train_s, y_train)
y_pred_dt = dt.predict(X_test_s)
y_prob_dt = dt.predict_proba(X_test_s)[:, 1]

acc_dt  = accuracy_score(y_test, y_pred_dt)
prec_dt = precision_score(y_test, y_pred_dt)
rec_dt  = recall_score(y_test, y_pred_dt)
f1_dt   = f1_score(y_test, y_pred_dt)
auc_dt  = roc_auc_score(y_test, y_prob_dt)

print("DECISION TREE CLASSIFIER")
print("-" * 40)
print(f"  Accuracy       : {acc_dt*100:.2f}%")
print(f"  Precision      : {prec_dt*100:.2f}%")
print(f"  Recall         : {rec_dt*100:.2f}%")
print(f"  F1-Score       : {f1_dt*100:.2f}%")
print(f"  ROC-AUC        : {auc_dt:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 1: Confusion Matrix (side by side)
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, y_pred, model_name, color in [
        (axes[0], y_pred_lr, "Logistic Regression", "Blues"),
        (axes[1], y_pred_dt, "Decision Tree",       "Greens")]:
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap=color, ax=ax,
                xticklabels=["Pred: Fail","Pred: Pass"],
                yticklabels=["Actual: Fail","Actual: Pass"],
                annot_kws={"size": 16, "weight": "bold"},
                linewidths=1, cbar=False)
    tn, fp, fn, tp = cm.ravel()
    ax.set_title(f"{model_name}\n"
                 f"TP={tp}  FP={fp}  FN={fn}  TN={tn}",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=10)
    ax.set_ylabel("True Label", fontsize=10)

plt.suptitle("Confusion Matrix Comparison", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/model/confusion_matrix.png", bbox_inches="tight")
plt.close()
print("✅  Saved → outputs/model/confusion_matrix.png")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 2: ROC Curves
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
for y_prob, label, color, auc in [
        (y_prob_lr, "Logistic Regression", "#1F4E79", auc_lr),
        (y_prob_dt, "Decision Tree",        "#70AD47", auc_dt)]:
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    ax.plot(fpr, tpr, lw=2.5, color=color, label=f"{label} (AUC={auc:.3f})")
ax.plot([0,1],[0,1], "k--", lw=1.5, label="Random Classifier (AUC=0.5)")
ax.fill_between(*roc_curve(y_test, y_prob_lr)[:2],
                alpha=0.08, color="#1F4E79")
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate (Recall)", fontsize=12)
ax.set_title("ROC Curve — Model Comparison", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.02])
plt.tight_layout()
plt.savefig("outputs/model/roc_curve.png", bbox_inches="tight")
plt.close()
print("✅  Saved → outputs/model/roc_curve.png")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 3: Precision-Recall Curve
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
for y_prob, label, color in [
        (y_prob_lr, "Logistic Regression", "#1F4E79"),
        (y_prob_dt, "Decision Tree",        "#70AD47")]:
    prec_c, rec_c, _ = precision_recall_curve(y_test, y_prob)
    ax.plot(rec_c, prec_c, lw=2.5, color=color, label=label)
ax.set_xlabel("Recall", fontsize=12)
ax.set_ylabel("Precision", fontsize=12)
ax.set_title("Precision-Recall Curve", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.02])
plt.tight_layout()
plt.savefig("outputs/model/precision_recall_curve.png", bbox_inches="tight")
plt.close()
print("✅  Saved → outputs/model/precision_recall_curve.png")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 4: Feature Importance
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# LR coefficients
lr_coef = pd.Series(np.abs(lr.coef_[0]), index=features).sort_values(ascending=True)
bars = axes[0].barh(lr_coef.index, lr_coef.values,
                    color=["#1F4E79" if v == lr_coef.max() else "#2E75B6"
                           for v in lr_coef.values])
axes[0].set_title("Logistic Regression\nFeature Importance (|Coefficient|)",
                  fontsize=11, fontweight="bold")
axes[0].set_xlabel("Absolute Coefficient Value", fontsize=10)
for bar, val in zip(bars, lr_coef.values):
    axes[0].text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                 f"{val:.3f}", va="center", fontsize=9)

# DT feature importance
dt_imp = pd.Series(dt.feature_importances_, index=features).sort_values(ascending=True)
bars = axes[1].barh(dt_imp.index, dt_imp.values,
                    color=["#70AD47" if v == dt_imp.max() else "#A9D18E"
                           for v in dt_imp.values])
axes[1].set_title("Decision Tree\nFeature Importance (Gini Impurity)",
                  fontsize=11, fontweight="bold")
axes[1].set_xlabel("Feature Importance Score", fontsize=10)
for bar, val in zip(bars, dt_imp.values):
    axes[1].text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                 f"{val:.3f}", va="center", fontsize=9)

plt.suptitle("Feature Importance Analysis", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/model/feature_importance.png", bbox_inches="tight")
plt.close()
print("✅  Saved → outputs/model/feature_importance.png")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 5: Model Performance Scorecard
# ─────────────────────────────────────────────────────────────────────────────
metrics = ["Accuracy","Precision","Recall","F1-Score","ROC-AUC"]
lr_scores = [acc_lr, prec_lr, rec_lr, f1_lr, auc_lr]
dt_scores = [acc_dt, prec_dt, rec_dt, f1_dt, auc_dt]

x = np.arange(len(metrics))
width = 0.35
fig, ax = plt.subplots(figsize=(11, 6))
bars1 = ax.bar(x - width/2, [s*100 for s in lr_scores], width, color="#1F4E79",
               label="Logistic Regression", edgecolor="white")
bars2 = ax.bar(x + width/2, [s*100 for s in dt_scores], width, color="#70AD47",
               label="Decision Tree", edgecolor="white")
for bars in [bars1, bars2]:
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}%", ha="center", va="bottom",
                fontsize=9, fontweight="bold")
ax.set_ylabel("Score (%)", fontsize=12)
ax.set_title("Model Performance Scorecard\n(Logistic Regression vs Decision Tree)",
             fontsize=13, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=11)
ax.set_ylim(0, 110)
ax.axhline(80, color="gray", linestyle="--", lw=1, alpha=0.5, label="80% benchmark")
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig("outputs/model/model_scorecard.png", bbox_inches="tight")
plt.close()
print("✅  Saved → outputs/model/model_scorecard.png")

# ─────────────────────────────────────────────────────────────────────────────
# Generate At-Risk Student Report
# ─────────────────────────────────────────────────────────────────────────────
df_all = df.copy()
X_all_s = scaler.transform(df_all[features])
df_all["Risk_Score"]    = lr.predict_proba(X_all_s)[:, 0]   # prob of FAIL
df_all["Risk_Category"] = pd.cut(df_all["Risk_Score"],
    bins=[0, 0.25, 0.50, 0.75, 1.0],
    labels=["Low Risk","Moderate Risk","High Risk","Critical Risk"])

at_risk_report = df_all[df_all["Risk_Score"] > 0.50][[
    "Student_ID","Department","Year","Semester",
    "LMS_Hours_Week","Attendance_Pct","Assignment_Score","Mid_Grade",
    "Final_Grade","Pass_Fail","Risk_Score","Risk_Category"
]].sort_values("Risk_Score", ascending=False)

at_risk_report.to_csv("outputs/model/at_risk_students_report.csv", index=False)
print(f"\n✅  At-Risk Report saved → outputs/model/at_risk_students_report.csv")
print(f"    Total at-risk students flagged : {len(at_risk_report)}")
print(f"    Critical Risk  : {(at_risk_report['Risk_Category']=='Critical Risk').sum()}")
print(f"    High Risk      : {(at_risk_report['Risk_Category']=='High Risk').sum()}")
print(f"    Moderate Risk  : {(at_risk_report['Risk_Category']=='Moderate Risk').sum()}")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("MODEL PERFORMANCE SUMMARY")
print("=" * 65)
summary = pd.DataFrame({
    "Metric":   metrics,
    "Logistic Regression (%)": [f"{s*100:.2f}" for s in lr_scores],
    "Decision Tree (%)":       [f"{s*100:.2f}" for s in dt_scores]
})
print(summary.to_string(index=False))
summary.to_csv("outputs/model/model_summary.csv", index=False)
print("\n✅  Summary saved → outputs/model/model_summary.csv")
print("=" * 65)

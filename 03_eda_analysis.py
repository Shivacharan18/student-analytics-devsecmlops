"""
=============================================================
SCRIPT 03 — Exploratory Data Analysis (EDA)
Case Study: Interactive Data Analytics and Visualization
           of Student Academic Performance
=============================================================
Tasks covered:
  • Descriptive statistics (mean, median, std, skewness)
  • t-test: High vs Low attendance cohorts
  • Pearson correlation matrix
  • Distribution histograms
  • Scatter plots (LMS_Hours vs Final_Grade)
  • Box plots by Department
  • All charts saved to outputs/eda/
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
import os, warnings
warnings.filterwarnings("ignore")

os.makedirs("outputs/eda", exist_ok=True)

# ── Styling ───────────────────────────────────────────────────────────────────
PALETTE   = ["#1F4E79", "#2E75B6", "#70AD47", "#ED7D31", "#A5A5A5"]
DEPT_CLR  = {"Computer Science":"#1F4E79","Electronics":"#2E75B6",
             "Business Admin":"#ED7D31","Mechanical":"#70AD47","Civil":"#A5A5A5"}
plt.rcParams.update({"font.family":"DejaVu Sans", "axes.spines.top":False,
                     "axes.spines.right":False, "figure.dpi":120})

df = pd.read_csv("data/student_performance_clean.csv")
num_cols = ["Age","Study_Hours_Week","LMS_Hours_Week","Attendance_Pct",
            "Assignment_Score","Mid_Grade","Final_Grade"]

# ── 1. Descriptive Statistics ─────────────────────────────────────────────────
print("=" * 65)
print("DESCRIPTIVE STATISTICS")
print("=" * 65)
desc = df[num_cols].agg(["mean","median","std","min","max",
                          lambda x: x.skew()]).round(3)
desc.index = ["Mean","Median","Std Dev","Min","Max","Skewness"]
print(desc.to_string())
desc.to_csv("outputs/eda/descriptive_statistics.csv")
print("\n✅  Saved → outputs/eda/descriptive_statistics.csv")

# Department-wise mean Final_Grade
dept_stats = df.groupby("Department")["Final_Grade"].agg(["mean","median","std","count"]).round(2)
dept_stats.columns = ["Mean","Median","Std Dev","Count"]
print("\nDepartment-wise Final Grade Stats:")
print(dept_stats.to_string())
dept_stats.to_csv("outputs/eda/dept_grade_stats.csv")

# ── 2. t-test: High vs Low Attendance ────────────────────────────────────────
print("\n" + "=" * 65)
print("t-TEST: High Attendance (≥75%) vs Low Attendance (<75%)")
print("=" * 65)
high_att = df[df["Attendance_Pct"] >= 75]["Final_Grade"]
low_att  = df[df["Attendance_Pct"] <  75]["Final_Grade"]
t_stat, p_val = stats.ttest_ind(high_att, low_att)
print(f"  High-Att mean : {high_att.mean():.2f}  (n={len(high_att)})")
print(f"  Low-Att  mean : {low_att.mean():.2f}   (n={len(low_att)})")
print(f"  t-statistic   : {t_stat:.4f}")
print(f"  p-value       : {p_val:.6f}  {'*** Significant' if p_val < 0.05 else 'Not significant'}")

# ── 3. Pearson Correlation Matrix ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("PEARSON CORRELATION MATRIX")
print("=" * 65)
corr_cols = ["Study_Hours_Week","LMS_Hours_Week","Attendance_Pct",
             "Assignment_Score","Mid_Grade","Final_Grade"]
corr_mat = df[corr_cols].corr(method="pearson").round(3)
print(corr_mat.to_string())
corr_mat.to_csv("outputs/eda/correlation_matrix.csv")

fig, ax = plt.subplots(figsize=(9, 7))
mask = np.triu(np.ones_like(corr_mat, dtype=bool))
sns.heatmap(corr_mat, annot=True, fmt=".2f", cmap="Blues",
            mask=mask, linewidths=0.5, ax=ax,
            annot_kws={"size": 11},
            cbar_kws={"shrink": 0.8})
ax.set_title("Pearson Correlation Matrix — Student Performance Features",
             fontsize=13, fontweight="bold", pad=15)
plt.tight_layout()
plt.savefig("outputs/eda/correlation_heatmap.png", bbox_inches="tight")
plt.close()
print("✅  Saved → outputs/eda/correlation_heatmap.png")

# ── 4. Final Grade Distribution Histogram ────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Full distribution
axes[0].hist(df["Final_Grade"], bins=25, color="#2E75B6", edgecolor="white",
             linewidth=0.6, alpha=0.85)
axes[0].axvline(df["Final_Grade"].mean(),   color="#ED7D31", lw=2,
                linestyle="--", label=f"Mean={df['Final_Grade'].mean():.1f}")
axes[0].axvline(df["Final_Grade"].median(), color="#70AD47", lw=2,
                linestyle="-.", label=f"Median={df['Final_Grade'].median():.1f}")
axes[0].axvline(50, color="red", lw=1.5, linestyle=":", label="Pass threshold (50)")
axes[0].set_xlabel("Final Grade", fontsize=11)
axes[0].set_ylabel("Number of Students", fontsize=11)
axes[0].set_title("Final Grade Distribution", fontsize=12, fontweight="bold")
axes[0].legend(fontsize=9)

# By department
for dept, color in DEPT_CLR.items():
    subset = df[df["Department"] == dept]["Final_Grade"]
    axes[1].hist(subset, bins=20, alpha=0.55, color=color,
                 edgecolor="white", linewidth=0.4, label=dept)
axes[1].axvline(50, color="red", lw=1.5, linestyle=":", label="Pass threshold")
axes[1].set_xlabel("Final Grade", fontsize=11)
axes[1].set_ylabel("Number of Students", fontsize=11)
axes[1].set_title("Final Grade Distribution by Department", fontsize=12, fontweight="bold")
axes[1].legend(fontsize=8, loc="upper left")

plt.suptitle("Distribution Analysis — Final Grades", fontsize=14,
             fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("outputs/eda/grade_distribution.png", bbox_inches="tight")
plt.close()
print("✅  Saved → outputs/eda/grade_distribution.png")

# ── 5. Scatter: LMS Hours vs Final Grade ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
for dept, color in DEPT_CLR.items():
    sub = df[df["Department"] == dept]
    ax.scatter(sub["LMS_Hours_Week"], sub["Final_Grade"],
               c=color, alpha=0.45, s=35, label=dept)

# Regression line
x = df["LMS_Hours_Week"].values
y = df["Final_Grade"].values
m, b, r, p, se = stats.linregress(x, y)
x_line = np.linspace(x.min(), x.max(), 100)
ax.plot(x_line, m * x_line + b, color="black", lw=2,
        label=f"y = {m:.2f}x + {b:.2f}  (r={r:.2f}, p<0.001)")
ax.axhline(50, color="red", lw=1.5, linestyle="--", alpha=0.6, label="Pass threshold")
ax.set_xlabel("LMS Hours / Week", fontsize=12)
ax.set_ylabel("Final Grade", fontsize=12)
ax.set_title("LMS Engagement vs Final Grade  (Scatter + Regression Line)",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("outputs/eda/scatter_lms_vs_grade.png", bbox_inches="tight")
plt.close()
print("✅  Saved → outputs/eda/scatter_lms_vs_grade.png")
print(f"   Pearson r={r:.3f}, p={p:.2e}")

# ── 6. Box Plot: Final Grade by Department ────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 6))
dept_order = df.groupby("Department")["Final_Grade"].median().sort_values(ascending=False).index.tolist()
bplot = ax.boxplot(
    [df[df["Department"] == d]["Final_Grade"].values for d in dept_order],
    tick_labels=dept_order, patch_artist=True, notch=False,
    medianprops={"color": "black", "linewidth": 2},
    whiskerprops={"linewidth": 1.5},
    flierprops={"marker": "o", "markersize": 4, "alpha": 0.4}
)
for patch, color in zip(bplot["boxes"], [DEPT_CLR[d] for d in dept_order]):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.axhline(50, color="red", linestyle="--", lw=1.5, label="Pass threshold (50)")
ax.set_xlabel("Department", fontsize=12)
ax.set_ylabel("Final Grade", fontsize=12)
ax.set_title("Final Grade Distribution by Department (Box Plot)",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig("outputs/eda/boxplot_dept_grade.png", bbox_inches="tight")
plt.close()
print("✅  Saved → outputs/eda/boxplot_dept_grade.png")

# ── 7. LMS Category vs Mean Final Grade Bar Chart ────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
lms_order = ["Low (<3 hrs)", "Medium (3-7 hrs)", "High (>7 hrs)"]
lms_means = df.groupby("LMS_Category")["Final_Grade"].mean().reindex(lms_order)
lms_counts= df.groupby("LMS_Category")["Final_Grade"].count().reindex(lms_order)
bars = ax.bar(lms_order, lms_means, color=["#C00000", "#ED7D31", "#70AD47"],
              edgecolor="white", width=0.55)
ax.axhline(50, color="red", linestyle="--", lw=1.5, alpha=0.7, label="Pass threshold")
for bar, mean_val, n in zip(bars, lms_means, lms_counts):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.2, f"{mean_val:.1f}\n(n={n})",
            ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_xlabel("LMS Engagement Category", fontsize=12)
ax.set_ylabel("Mean Final Grade", fontsize=12)
ax.set_title("Impact of LMS Engagement on Final Grade",
             fontsize=13, fontweight="bold")
ax.set_ylim(0, 100)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig("outputs/eda/lms_category_vs_grade.png", bbox_inches="tight")
plt.close()
print("✅  Saved → outputs/eda/lms_category_vs_grade.png")

print("\n" + "=" * 65)
print("✅  EDA complete. All outputs in outputs/eda/")
print("=" * 65)

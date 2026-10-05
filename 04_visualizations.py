"""
=============================================================
SCRIPT 04 — Advanced Visualizations
Case Study: Interactive Data Analytics and Visualization
           of Student Academic Performance
=============================================================
Tasks covered:
  • Performance Failure Heatmap (Week × Department)
  • Student Journey Sankey Diagram
  • Weekly Performance Trend Lines (16 Weeks)
  • Pass/Fail stacked bar by Department + Year
  • All charts saved to outputs/visualizations/
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import seaborn as sns
import os, warnings
warnings.filterwarnings("ignore")

os.makedirs("outputs/visualizations", exist_ok=True)

plt.rcParams.update({"font.family": "DejaVu Sans",
                     "axes.spines.top": False,
                     "axes.spines.right": False,
                     "figure.dpi": 120})

DEPT_CLR = {"Computer Science":"#1F4E79","Electronics":"#2E75B6",
            "Business Admin":"#ED7D31","Mechanical":"#70AD47","Civil":"#A5A5A5"}

df = pd.read_csv("data/student_performance_clean.csv")

# ── 1. Performance Failure Heatmap (Week × Department) ───────────────────────
print("Building Failure Heatmap...")

# Parse weekly performance
weekly_df = df["Weekly_Performance"].str.split(";", expand=True).astype(float)
weekly_df.columns = [f"W{w+1}" for w in range(16)]
weekly_df["Department"] = df["Department"].values
weekly_df["Student_ID"]  = df["Student_ID"].values

# Compute % below 50 per week per department
departments = list(DEPT_CLR.keys())
weeks       = [f"W{w+1}" for w in range(16)]
failure_pct = pd.DataFrame(index=departments, columns=weeks, dtype=float)

for dept in departments:
    sub = weekly_df[weekly_df["Department"] == dept][weeks]
    failure_pct.loc[dept] = (sub < 50).mean() * 100

fig, ax = plt.subplots(figsize=(14, 5))
sns.heatmap(failure_pct.astype(float), ax=ax, cmap="YlOrRd",
            annot=True, fmt=".0f", linewidths=0.4,
            annot_kws={"size": 8},
            cbar_kws={"label": "% Students with Grade < 50", "shrink": 0.85})

# Mark critical dip zones
for spine in ["top","bottom","left","right"]:
    ax.spines[spine].set_visible(False)
ax.set_title("Performance Failure Heatmap\n(% of Students with Weekly Grade < 50, by Department × Week)",
             fontsize=12, fontweight="bold", pad=12)
ax.set_xlabel("Academic Week", fontsize=11)
ax.set_ylabel("Department", fontsize=11)
ax.axvline(4,  color="#1F4E79", lw=2.5, linestyle="--", alpha=0.6)
ax.axvline(7,  color="#1F4E79", lw=2.5, linestyle="--", alpha=0.6)
ax.axvline(10, color="#2E75B6", lw=2.5, linestyle=":",  alpha=0.6)
ax.axvline(13, color="#2E75B6", lw=2.5, linestyle=":",  alpha=0.6)

# Annotations for dip zones
ax.text(5.5, -0.45, "Pre-Midterm\nDip Zone", ha="center", fontsize=8,
        color="#1F4E79", fontweight="bold")
ax.text(11.5, -0.45, "Pre-Final\nDip Zone", ha="center", fontsize=8,
        color="#2E75B6", fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/visualizations/failure_heatmap.png", bbox_inches="tight")
plt.close()
print("✅  Saved → outputs/visualizations/failure_heatmap.png")

# ── 2. Student Journey Sankey (Matplotlib approximation) ─────────────────────
print("Building Student Journey Flow Chart...")

def grade_band_entry(s):
    if s < 50:   return "Low Entry"
    elif s < 70: return "Mid Entry"
    else:        return "High Entry"

def grade_band_mid(s):
    if s < 50:   return "Low Mid"
    elif s < 70: return "Mid Semester"
    else:        return "High Mid"

df["Entry_Band"]  = df["Assignment_Score"].apply(grade_band_entry)
df["Mid_Band"]    = df["Mid_Grade"].apply(grade_band_mid)
df["Final_Outcome"] = df["Pass_Fail"].map({1: "PASS", 0: "FAIL"})

# Count flows: Entry → Mid → Outcome
flow_data = df.groupby(["Entry_Band","Mid_Band","Final_Outcome"]).size().reset_index(name="Count")

# Simplified Sankey using stacked bar flow diagram
fig, axes = plt.subplots(1, 3, figsize=(14, 7))
fig.suptitle("Student Academic Journey Flow\n(Entry Performance → Mid-Semester → Final Outcome)",
             fontsize=13, fontweight="bold")

stage_colors = {
    "Low Entry":"#C00000","Mid Entry":"#ED7D31","High Entry":"#70AD47",
    "Low Mid":"#C00000","Mid Semester":"#ED7D31","High Mid":"#70AD47",
    "PASS":"#1F4E79","FAIL":"#C00000"
}

for ax_idx, (stage, col) in enumerate([
        ("Entry_Band", "Entry Performance (Assignment Score)"),
        ("Mid_Band",   "Mid-Semester Performance"),
        ("Final_Outcome", "Final Outcome")]):
    counts = df[stage].value_counts()
    colors = [stage_colors.get(k, "#888888") for k in counts.index]
    bars = axes[ax_idx].barh(counts.index, counts.values, color=colors, edgecolor="white")
    for bar, val in zip(bars, counts.values):
        pct = val / len(df) * 100
        axes[ax_idx].text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
                          f"{val} ({pct:.1f}%)", va="center", fontsize=9)
    axes[ax_idx].set_title(col, fontsize=10, fontweight="bold")
    axes[ax_idx].set_xlabel("Number of Students", fontsize=9)
    axes[ax_idx].set_xlim(0, counts.max() * 1.3)

plt.tight_layout()
plt.savefig("outputs/visualizations/student_journey_sankey.png", bbox_inches="tight")
plt.close()

# Critical path analysis
low_low_fail = len(df[(df["Entry_Band"] == "Low Entry") &
                      (df["Mid_Band"] == "Low Mid") &
                      (df["Final_Outcome"] == "FAIL")])
low_low_total = len(df[(df["Entry_Band"] == "Low Entry") & (df["Mid_Band"] == "Low Mid")])
pct = low_low_fail / low_low_total * 100 if low_low_total > 0 else 0
print(f"✅  Saved → outputs/visualizations/student_journey_sankey.png")
print(f"   Critical path (Low→Low→Fail): {pct:.1f}% of Low-Low students fail")

# ── 3. Weekly Performance Trend Lines by Department ─────────────────────────
print("Building Weekly Trend Lines...")

fig, ax = plt.subplots(figsize=(14, 6))
for dept, color in DEPT_CLR.items():
    sub  = weekly_df[weekly_df["Department"] == dept][weeks].astype(float)
    mean = sub.mean()
    ci   = sub.std() / np.sqrt(len(sub)) * 1.96   # 95% CI
    ax.plot(range(1, 17), mean, color=color, lw=2.2, label=dept, marker="o", markersize=4)
    ax.fill_between(range(1, 17), mean - ci, mean + ci,
                    color=color, alpha=0.12)

ax.axhline(50, color="red", lw=1.5, linestyle="--", alpha=0.7, label="Pass threshold")
ax.axvspan(4.5, 7.5,  color="#FF9999", alpha=0.12, label="Pre-Midterm Dip Zone")
ax.axvspan(10.5, 13.5, color="#9999FF", alpha=0.12, label="Pre-Final Dip Zone")
ax.set_xlabel("Academic Week", fontsize=12)
ax.set_ylabel("Mean Weekly Performance Score", fontsize=12)
ax.set_title("16-Week Performance Trend Lines by Department\n(Shaded bands = 95% Confidence Interval)",
             fontsize=13, fontweight="bold")
ax.set_xticks(range(1, 17))
ax.set_xticklabels([f"W{w}" for w in range(1, 17)], fontsize=8)
ax.legend(fontsize=9, loc="lower right")
ax.set_ylim(30, 90)
plt.tight_layout()
plt.savefig("outputs/visualizations/weekly_trend_lines.png", bbox_inches="tight")
plt.close()
print("✅  Saved → outputs/visualizations/weekly_trend_lines.png")

# ── 4. Pass/Fail Stacked Bar by Department + Year ────────────────────────────
print("Building Pass/Fail Stacked Bar Charts...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# By Department
pf_dept = df.groupby(["Department","Pass_Fail"]).size().unstack(fill_value=0)
pf_dept.columns = ["Fail","Pass"]
pf_dept_pct = pf_dept.div(pf_dept.sum(axis=1), axis=0) * 100
pf_dept_pct[["Pass","Fail"]].plot(kind="bar", stacked=True, ax=axes[0],
    color=["#1F4E79","#C00000"], edgecolor="white", rot=25)
axes[0].set_title("Pass vs Fail Rate by Department", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Department", fontsize=10)
axes[0].set_ylabel("Percentage of Students (%)", fontsize=10)
axes[0].legend(["Pass","Fail"], fontsize=9)
for container in axes[0].containers:
    axes[0].bar_label(container, fmt="%.1f%%", label_type="center",
                      fontsize=8, color="white", fontweight="bold")

# By Year
pf_year = df.groupby(["Year","Pass_Fail"]).size().unstack(fill_value=0)
pf_year.columns = ["Fail","Pass"]
pf_year_pct = pf_year.div(pf_year.sum(axis=1), axis=0) * 100
pf_year_pct[["Pass","Fail"]].plot(kind="bar", stacked=True, ax=axes[1],
    color=["#2E75B6","#ED7D31"], edgecolor="white", rot=0)
axes[1].set_title("Pass vs Fail Rate by Academic Year", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Year", fontsize=10)
axes[1].set_ylabel("Percentage of Students (%)", fontsize=10)
axes[1].legend(["Pass","Fail"], fontsize=9)
for container in axes[1].containers:
    axes[1].bar_label(container, fmt="%.1f%%", label_type="center",
                      fontsize=8, color="white", fontweight="bold")

plt.suptitle("Pass / Fail Analysis", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("outputs/visualizations/pass_fail_analysis.png", bbox_inches="tight")
plt.close()
print("✅  Saved → outputs/visualizations/pass_fail_analysis.png")

# ── 5. Attendance + LMS Combined Impact ──────────────────────────────────────
print("Building Attendance × LMS Impact Matrix...")

att_cat_order = ["Critical (<60%)","Low (60-75%)","Moderate (75-85%)","High (>85%)"]
lms_cat_order = ["Low (<3 hrs)","Medium (3-7 hrs)","High (>7 hrs)"]

pivot = df.groupby(["Attendance_Category","LMS_Category"])["Final_Grade"].mean().unstack()
pivot = pivot.reindex(index=att_cat_order, columns=lms_cat_order)

fig, ax = plt.subplots(figsize=(9, 6))
sns.heatmap(pivot, ax=ax, cmap="RdYlGn", annot=True, fmt=".1f",
            linewidths=0.5, vmin=30, vmax=90,
            cbar_kws={"label": "Mean Final Grade", "shrink": 0.85},
            annot_kws={"size": 13, "weight": "bold"})
ax.set_title("Mean Final Grade: Attendance Category × LMS Engagement\n(Red = At Risk, Green = Strong Performance)",
             fontsize=12, fontweight="bold", pad=12)
ax.set_xlabel("LMS Hours per Week", fontsize=11)
ax.set_ylabel("Attendance Category", fontsize=11)
plt.tight_layout()
plt.savefig("outputs/visualizations/attendance_lms_matrix.png", bbox_inches="tight")
plt.close()
print("✅  Saved → outputs/visualizations/attendance_lms_matrix.png")

print("\n" + "=" * 65)
print("✅  All advanced visualizations complete. See outputs/visualizations/")
print("=" * 65)

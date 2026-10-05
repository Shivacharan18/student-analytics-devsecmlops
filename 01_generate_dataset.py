"""
=============================================================
SCRIPT 01 — Dataset Generation
Case Study: Interactive Data Analytics and Visualization
           of Student Academic Performance
=============================================================
Generates a realistic Kaggle-style student performance CSV
dataset (1,044 student records) with all features referenced
in the case study report.
=============================================================
"""

import pandas as pd
import numpy as np
import os

np.random.seed(42)
N = 1044

# ── Helpers ──────────────────────────────────────────────────────────────────
def clamp(arr, lo=0, hi=100):
    return np.clip(arr, lo, hi)

# ── Demographics ──────────────────────────────────────────────────────────────
student_ids   = [f"STU{1000 + i}" for i in range(N)]
ages          = np.random.randint(17, 26, N)
genders       = np.random.choice(["Male", "Female"], N, p=[0.52, 0.48])
departments   = np.random.choice(
    ["Computer Science", "Electronics", "Business Admin", "Mechanical", "Civil"],
    N, p=[0.28, 0.22, 0.20, 0.18, 0.12]
)
years         = np.random.choice([1, 2, 3, 4], N, p=[0.30, 0.28, 0.24, 0.18])
semesters     = np.where(np.random.rand(N) < 0.5, "Sem-1", "Sem-2")
parental_edu  = np.random.choice(
    ["Below HSC", "HSC", "Graduate", "Post-Graduate"], N, p=[0.15, 0.25, 0.40, 0.20]
)
internet      = np.random.choice(["Yes", "No"], N, p=[0.80, 0.20])

# ── Behavioural variables (correlated with grade) ────────────────────────────
study_hours   = np.random.normal(8, 4, N).clip(1, 30).round(1)
lms_hours     = (study_hours * 0.55 + np.random.normal(0, 1.5, N)).clip(0.5, 18).round(1)
attendance    = clamp(70 + study_hours * 1.2 + np.random.normal(0, 8, N), 30, 100).round(1)

# Assignment score — strong predictor
assignment    = clamp(45 + study_hours * 2.0 + lms_hours * 1.5
                      + np.random.normal(0, 8, N), 20, 100).round(1)

# Midterm — depends on assignment + attendance
mid_grade     = clamp(assignment * 0.55 + attendance * 0.20
                      + np.random.normal(0, 7, N), 10, 100).round(1)

# Final — strongest model
final_grade   = clamp(assignment * 0.40 + lms_hours * 1.8
                      + attendance * 0.25 + np.random.normal(0, 6, N), 12, 100).round(1)

pass_fail     = (final_grade >= 50).astype(int)

# At-Risk flag: grade dropped >20% from expected trajectory
expected_final = clamp(mid_grade * 1.05, 0, 100)
at_risk        = ((expected_final - final_grade) > 20).astype(int)

# Inject ~4% missing values realistically
def inject_missing(arr, pct=0.04):
    out = arr.astype(object).copy()
    idx = np.random.choice(len(arr), int(len(arr) * pct), replace=False)
    out[idx] = np.nan
    return out

lms_hours_m   = inject_missing(lms_hours, 0.042)
attendance_m  = inject_missing(attendance, 0.018)
parental_edu_m= inject_missing(parental_edu, 0.065)

# Weekly performance proxy (16 weeks) — stored as comma-separated string
def weekly_scores(final, n_weeks=16):
    base     = final * 0.8
    scores   = []
    current  = base + np.random.normal(0, 5)
    for w in range(n_weeks):
        # Dip at weeks 5-7 and 11-13 (pre-exam stress)
        noise = np.random.normal(-3, 4) if w in [4, 5, 6, 10, 11, 12] else np.random.normal(1, 3)
        current = clamp(np.array([current + noise]), 0, 100)[0]
        scores.append(round(current, 1))
    return ";".join(map(str, scores))

weekly_perf = [weekly_scores(fg) for fg in final_grade]

# ── Build DataFrame ───────────────────────────────────────────────────────────
df = pd.DataFrame({
    "Student_ID":          student_ids,
    "Age":                 ages,
    "Gender":              genders,
    "Department":          departments,
    "Year":                years,
    "Semester":            semesters,
    "Parental_Education":  parental_edu_m,
    "Internet_Access":     internet,
    "Study_Hours_Week":    study_hours,
    "LMS_Hours_Week":      lms_hours_m,
    "Attendance_Pct":      attendance_m,
    "Assignment_Score":    assignment,
    "Mid_Grade":           mid_grade,
    "Final_Grade":         final_grade,
    "Pass_Fail":           pass_fail,
    "At_Risk_Flag":        at_risk,
    "Weekly_Performance":  weekly_perf,
})

# Inject 12 exact duplicate rows (for cleaning demo)
dup_idx = np.random.choice(N, 12, replace=False)
df = pd.concat([df, df.iloc[dup_idx]], ignore_index=True)

os.makedirs("data", exist_ok=True)
df.to_csv("data/student_performance_raw.csv", index=False)

print(f"✅  Raw dataset saved → data/student_performance_raw.csv")
print(f"    Rows : {len(df):,}  |  Columns : {df.shape[1]}")
print(f"    Pass rate   : {pass_fail.mean()*100:.1f}%")
print(f"    At-Risk     : {at_risk.sum()} students ({at_risk.mean()*100:.1f}%)")
print(f"    Missing LMS : {pd.isna(lms_hours_m).sum()} rows")

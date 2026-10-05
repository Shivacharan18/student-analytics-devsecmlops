"""
=============================================================
SCRIPT 02 — Data Cleaning & Preprocessing
Case Study: Interactive Data Analytics and Visualization
           of Student Academic Performance
=============================================================
Tasks covered:
  • Duplicate removal
  • Missing value imputation (median / mode)
  • Grade normalization (min-max)
  • Feature engineering (At_Risk_Flag, LMS_Category)
  • Saves cleaned master dataset
=============================================================
"""

import pandas as pd
import numpy as np
import os

# ── Load raw dataset ──────────────────────────────────────────────────────────
df = pd.read_csv("data/student_performance_raw.csv")
print("=" * 60)
print("STEP 1  — Raw Dataset Overview")
print("=" * 60)
print(f"Shape          : {df.shape}")
print(f"Columns        : {list(df.columns)}\n")
print(df.dtypes)
print("\nFirst 3 rows:")
print(df.head(3).to_string())

# ── 1. Duplicate Removal ──────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2  — Duplicate Removal")
print("=" * 60)
dupes = df.duplicated().sum()
print(f"Duplicate rows found : {dupes}")
df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)
print(f"Rows after removal   : {len(df)}")

# ── 2. Missing Value Analysis ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3  — Missing Value Analysis")
print("=" * 60)
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
mv_report = pd.DataFrame({"Missing Count": missing, "Missing %": missing_pct})
print(mv_report[mv_report["Missing Count"] > 0].to_string())

# Impute numerical columns with median
for col in ["LMS_Hours_Week", "Attendance_Pct"]:
    median_val = df[col].median()
    df[col] = df[col].fillna(median_val)
    print(f"\n  ✅ '{col}' → imputed with median = {median_val:.2f}")

# Impute categorical column with mode
mode_val = df["Parental_Education"].mode()[0]
df["Parental_Education"] = df["Parental_Education"].fillna(mode_val)
print(f"  ✅ 'Parental_Education' → imputed with mode = '{mode_val}'")

print(f"\nMissing values after imputation : {df.isnull().sum().sum()}")

# ── 3. Grade Normalization ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4  — Grade Normalization (Min-Max to 0-100 scale)")
print("=" * 60)
grade_cols = ["Assignment_Score", "Mid_Grade", "Final_Grade"]
for col in grade_cols:
    mn, mx = df[col].min(), df[col].max()
    df[f"{col}_Norm"] = ((df[col] - mn) / (mx - mn) * 100).round(2)
    print(f"  {col}: min={mn:.1f}, max={mx:.1f} → normalized range [0, 100]")

# ── 4. Feature Engineering ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5  — Feature Engineering")
print("=" * 60)

# LMS Category
def lms_category(h):
    if h < 3:   return "Low (<3 hrs)"
    elif h < 7: return "Medium (3-7 hrs)"
    else:       return "High (>7 hrs)"

df["LMS_Category"] = df["LMS_Hours_Week"].apply(lms_category)
print(f"  LMS_Category distribution:\n{df['LMS_Category'].value_counts().to_string()}")

# Attendance Category
def att_category(a):
    if a < 60:   return "Critical (<60%)"
    elif a < 75: return "Low (60-75%)"
    elif a < 85: return "Moderate (75-85%)"
    else:        return "High (>85%)"

df["Attendance_Category"] = df["Attendance_Pct"].apply(att_category)
print(f"\n  Attendance_Category distribution:\n{df['Attendance_Category'].value_counts().to_string()}")

# Grade Band
def grade_band(g):
    if g < 35:   return "Very Low (F)"
    elif g < 50: return "Low (D)"
    elif g < 65: return "Average (C)"
    elif g < 80: return "Good (B)"
    else:        return "Excellent (A)"

df["Grade_Band"] = df["Final_Grade"].apply(grade_band)

# Re-derive At_Risk_Flag cleanly
expected = (df["Mid_Grade"] * 1.05).clip(0, 100)
df["At_Risk_Flag"] = ((expected - df["Final_Grade"]) > 20).astype(int)
print(f"\n  At-Risk students flagged : {df['At_Risk_Flag'].sum()}")

# ── 5. Save cleaned dataset ───────────────────────────────────────────────────
os.makedirs("data", exist_ok=True)
df.to_csv("data/student_performance_clean.csv", index=False)
print("\n" + "=" * 60)
print(f"✅  Cleaned dataset saved → data/student_performance_clean.csv")
print(f"    Final shape : {df.shape}")
print("=" * 60)

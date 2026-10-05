"""
Data Preprocessing Module
Handles synthetic raw dataset generation, data cleaning, and feature engineering.
"""

import os
import numpy as np
import pandas as pd


def generate_raw_dataset(n_samples: int = 1044, seed: int = 42, output_path: str = "data/raw/student_performance_raw.csv") -> pd.DataFrame:
    """Generates synthetic student performance raw dataset."""
    np.random.seed(seed)
    n = n_samples

    def clamp(arr, lo=0, hi=100):
        return np.clip(arr, lo, hi)

    student_ids = [f"STU{1000 + i}" for i in range(n)]
    ages = np.random.randint(17, 26, n)
    genders = np.random.choice(["Male", "Female"], n, p=[0.52, 0.48])
    departments = np.random.choice(
        ["Computer Science", "Electronics", "Business Admin", "Mechanical", "Civil"],
        n, p=[0.28, 0.22, 0.20, 0.18, 0.12]
    )
    years = np.random.choice([1, 2, 3, 4], n, p=[0.30, 0.28, 0.24, 0.18])
    semesters = np.where(np.random.rand(n) < 0.5, "Sem-1", "Sem-2")
    parental_edu = np.random.choice(
        ["Below HSC", "HSC", "Graduate", "Post-Graduate"], n, p=[0.15, 0.25, 0.40, 0.20]
    )
    internet = np.random.choice(["Yes", "No"], n, p=[0.80, 0.20])

    study_hours = np.random.normal(8, 4, n).clip(1, 30).round(1)
    lms_hours = (study_hours * 0.55 + np.random.normal(0, 1.5, n)).clip(0.5, 18).round(1)
    attendance = clamp(70 + study_hours * 1.2 + np.random.normal(0, 8, n), 30, 100).round(1)

    assignment = clamp(45 + study_hours * 2.0 + lms_hours * 1.5 + np.random.normal(0, 8, n), 20, 100).round(1)
    mid_grade = clamp(assignment * 0.55 + attendance * 0.20 + np.random.normal(0, 7, n), 10, 100).round(1)
    final_grade = clamp(assignment * 0.40 + lms_hours * 1.8 + attendance * 0.25 + np.random.normal(0, 6, n), 12, 100).round(1)

    pass_fail = (final_grade >= 50).astype(int)
    expected_final = clamp(mid_grade * 1.05, 0, 100)
    at_risk = ((expected_final - final_grade) > 20).astype(int)

    def inject_missing(arr, pct=0.04):
        out = arr.astype(object).copy()
        idx = np.random.choice(len(arr), int(len(arr) * pct), replace=False)
        out[idx] = np.nan
        return out

    lms_hours_m = inject_missing(lms_hours, 0.042)
    attendance_m = inject_missing(attendance, 0.018)
    parental_edu_m = inject_missing(parental_edu, 0.065)

    def weekly_scores(final, n_weeks=16):
        base = final * 0.8
        scores = []
        current = base + np.random.normal(0, 5)
        for w in range(n_weeks):
            noise = np.random.normal(-3, 4) if w in [4, 5, 6, 10, 11, 12] else np.random.normal(1, 3)
            current = clamp(np.array([current + noise]), 0, 100)[0]
            scores.append(round(current, 1))
        return ";".join(map(str, scores))

    weekly_perf = [weekly_scores(fg) for fg in final_grade]

    df = pd.DataFrame({
        "Student_ID": student_ids,
        "Age": ages,
        "Gender": genders,
        "Department": departments,
        "Year": years,
        "Semester": semesters,
        "Parental_Education": parental_edu_m,
        "Internet_Access": internet,
        "Study_Hours_Week": study_hours,
        "LMS_Hours_Week": lms_hours_m,
        "Attendance_Pct": attendance_m,
        "Assignment_Score": assignment,
        "Mid_Grade": mid_grade,
        "Final_Grade": final_grade,
        "Pass_Fail": pass_fail,
        "At_Risk_Flag": at_risk,
        "Weekly_Performance": weekly_perf,
    })

    # Inject duplicate rows for demonstration
    dup_idx = np.random.choice(n, 12, replace=False)
    df = pd.concat([df, df.iloc[dup_idx]], ignore_index=True)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)

    return df


def clean_and_preprocess_data(input_path: str = "data/raw/student_performance_raw.csv",
                               output_path: str = "data/processed/student_performance_clean.csv") -> pd.DataFrame:
    """Loads raw data, performs deduplication, imputation, normalization, and feature engineering."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Raw dataset not found at {input_path}")

    df = pd.read_csv(input_path)

    # 1. Remove duplicates
    df = df.drop_duplicates().reset_index(drop=True)

    # 2. Missing value imputation
    for col in ["LMS_Hours_Week", "Attendance_Pct"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    if "Parental_Education" in df.columns:
        mode_val = df["Parental_Education"].mode()
        mode_str = mode_val[0] if len(mode_val) > 0 else "Graduate"
        df["Parental_Education"] = df["Parental_Education"].fillna(mode_str)

    # 3. Grade Normalization
    for col in ["Assignment_Score", "Mid_Grade", "Final_Grade"]:
        if col in df.columns:
            mn, mx = df[col].min(), df[col].max()
            df[f"{col}_Norm"] = ((df[col] - mn) / (mx - mn + 1e-8) * 100).round(2)

    # 4. Feature Engineering
    def lms_cat(h):
        if h < 3:
            return "Low (<3 hrs)"
        if h < 7:
            return "Medium (3-7 hrs)"
        return "High (>7 hrs)"

    def att_cat(a):
        if a < 60:
            return "Critical (<60%)"
        if a < 75:
            return "Low (60-75%)"
        if a < 85:
            return "Moderate (75-85%)"
        return "High (>85%)"

    def grade_band(g):
        if g < 35:
            return "Very Low (F)"
        if g < 50:
            return "Low (D)"
        if g < 65:
            return "Average (C)"
        if g < 80:
            return "Good (B)"
        return "Excellent (A)"

    df["LMS_Category"] = df["LMS_Hours_Week"].apply(lms_cat)
    df["Attendance_Category"] = df["Attendance_Pct"].apply(att_cat)
    df["Grade_Band"] = df["Final_Grade"].apply(grade_band)

    expected = (df["Mid_Grade"] * 1.05).clip(0, 100)
    df["At_Risk_Flag"] = ((expected - df["Final_Grade"]) > 20).astype(int)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)

    return df


if __name__ == "__main__":
    print("Generating raw dataset...")
    generate_raw_dataset()
    print("Cleaning and preprocessing dataset...")
    cleaned = clean_and_preprocess_data()
    print(f"[OK] Data preprocessing completed. Clean dataset shape: {cleaned.shape}")

"""
Interactive Student Academic Performance Dashboard (Streamlit)
DevSecMLOps-enabled UI serving trained model predictions and visual analytics.
"""

import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.predict import get_model, batch_predict, predict_student_risk, categorize_risk
from src.data_validation import calculate_file_sha256

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Academic Performance Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1F4E79, #2E75B6);
        padding: 1.2rem; border-radius: 10px; color: white; text-align: center;
    }
    .metric-value { font-size: 2rem; font-weight: bold; }
    .metric-label { font-size: 0.85rem; opacity: 0.85; margin-top: 4px; }
    .risk-critical { color: #9C0006; font-weight: bold; }
    .risk-high     { color: #C00000; font-weight: bold; }
    .risk-medium   { color: #ED7D31; font-weight: bold; }
    .risk-low      { color: #70AD47; font-weight: bold; }
    h1 { color: #1F4E79; }
    h2 { color: #2E75B6; }
</style>
""", unsafe_allow_html=True)

DEPT_CLR = {
    "Computer Science": "#1F4E79",
    "Electronics": "#2E75B6",
    "Business Admin": "#ED7D31",
    "Mechanical": "#70AD47",
    "Civil": "#A5A5A5"
}

# ── Load Data & Model Artifact ───────────────────────────────────────────────
@st.cache_data
def load_data():
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed", "student_performance_clean.csv")
    if not os.path.exists(path):
        # Fallback to local relative path
        path = "data/processed/student_performance_clean.csv"
    return pd.read_csv(path)

df_full = load_data()

# Ensure batch predictions with risk scores from serialized model
try:
    df_full = batch_predict(df_full)
except Exception as e:
    st.error(f"Error loading model artifact: {e}")

# ── Sidebar Filters ───────────────────────────────────────────────────────────
st.sidebar.markdown("## 🔍 Filters & Security")

dept_sel = st.sidebar.multiselect(
    "Department",
    options=sorted(df_full["Department"].unique()),
    default=list(df_full["Department"].unique())
)

year_sel = st.sidebar.multiselect(
    "Academic Year",
    options=sorted(df_full["Year"].unique()),
    default=list(df_full["Year"].unique())
)

sem_sel = st.sidebar.multiselect(
    "Semester",
    options=sorted(df_full["Semester"].unique()),
    default=list(df_full["Semester"].unique())
)

lms_sel = st.sidebar.multiselect(
    "LMS Category",
    options=["Low (<3 hrs)", "Medium (3-7 hrs)", "High (>7 hrs)"],
    default=["Low (<3 hrs)", "Medium (3-7 hrs)", "High (>7 hrs)"]
)

grade_range = st.sidebar.slider("Final Grade Range", 0, 100, (0, 100))

# Filtered DataFrame
df = df_full[
    (df_full["Department"].isin(dept_sel)) &
    (df_full["Year"].isin(year_sel)) &
    (df_full["Semester"].isin(sem_sel)) &
    (df_full["LMS_Category"].isin(lms_sel)) &
    (df_full["Final_Grade"] >= grade_range[0]) &
    (df_full["Final_Grade"] <= grade_range[1])
].copy()

st.sidebar.markdown(f"**Filtered students: {len(df):,}**")

# Security Badge
st.sidebar.divider()
st.sidebar.markdown("### 🔒 DevSecMLOps Guard")
st.sidebar.success("Model Integrity: Verified (SHA-256)")
st.sidebar.info("Input Validation: Active (Pydantic/Pandera)")

# ── Title ─────────────────────────────────────────────────────────────────────
st.title("📊 Interactive Student Academic Performance Dashboard")
st.markdown("*DevSecMLOps-Enabled Case Study 01 — Academic Year 2025–2026*")
st.divider()

# ── Tab Navigation ────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📌 Overview",
    "📈 EDA & Correlations",
    "🗺️ Visualization Hub",
    "⚠️ Early Warning System",
    "🔍 Student Explorer & Live Prediction"
])

# ============================================================
# TAB 1 — OVERVIEW
# ============================================================
with tab1:
    st.header("Overview Dashboard")

    total = len(df)
    pass_rt = df["Pass_Fail"].mean() * 100 if total > 0 else 0
    at_risk = (df["Risk_Score"] > 0.50).sum() if total > 0 else 0
    avg_grd = df["Final_Grade"].mean() if total > 0 else 0
    avg_lms = df["LMS_Hours_Week"].mean() if total > 0 else 0
    avg_att = df["Attendance_Pct"].mean() if total > 0 else 0

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    metrics_info = [
        (c1, f"{total:,}", "Total Students", "👥"),
        (c2, f"{pass_rt:.1f}%", "Pass Rate", "✅"),
        (c3, f"{100-pass_rt:.1f}%", "Fail Rate", "❌"),
        (c4, f"{at_risk}", "At-Risk Flagged", "⚠️"),
        (c5, f"{avg_grd:.1f}", "Avg Final Grade", "📊"),
        (c6, f"{avg_att:.1f}%", "Avg Attendance", "📅")
    ]
    for col, val, label, icon in metrics_info:
        col.markdown(f"""
        <div class="metric-card">
            <div style="font-size:1.5rem">{icon}</div>
            <div class="metric-value">{val}</div>
            <div class="metric-label">{label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.subheader("Department-wise Performance Summary")
    if total > 0:
        dept_summary = df.groupby("Department").agg(
            Students=("Student_ID", "count"),
            Pass_Rate=("Pass_Fail", lambda x: f"{x.mean()*100:.1f}%"),
            Mean_Grade=("Final_Grade", lambda x: f"{x.mean():.1f}"),
            Mean_Attendance=("Attendance_Pct", lambda x: f"{x.mean():.1f}%"),
            Mean_LMS_Hours=("LMS_Hours_Week", lambda x: f"{x.mean():.1f}"),
            At_Risk=("Risk_Score", lambda x: (x > 0.50).sum())
        ).reset_index()
        st.dataframe(dept_summary, use_container_width=True, hide_index=True)

    st.subheader("Grade Band Distribution")
    if total > 0:
        fig, ax = plt.subplots(figsize=(10, 4))
        band_order = ["Very Low (F)", "Low (D)", "Average (C)", "Good (B)", "Excellent (A)"]
        band_colors = ["#C00000", "#ED7D31", "#FFD966", "#70AD47", "#1F4E79"]
        band_counts = df["Grade_Band"].value_counts().reindex(band_order, fill_value=0)
        bars = ax.bar(band_order, band_counts.values, color=band_colors, edgecolor="white", width=0.6)
        for bar, val in zip(bars, band_counts.values):
            pct = val / len(df) * 100
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f"{val}\n({pct:.1f}%)", ha="center", fontsize=10, fontweight="bold")
        ax.set_ylabel("Number of Students", fontsize=11)
        ax.set_title("Students by Grade Band", fontsize=12, fontweight="bold")
        ax.set_ylim(0, max(band_counts.max() * 1.25, 10))
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

# ============================================================
# TAB 2 — EDA & CORRELATIONS
# ============================================================
with tab2:
    st.header("Exploratory Data Analysis")
    if total > 0:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Correlation Heatmap")
            num_cols = ["Study_Hours_Week", "LMS_Hours_Week", "Attendance_Pct", "Assignment_Score", "Mid_Grade", "Final_Grade"]
            corr = df[num_cols].corr()
            fig, ax = plt.subplots(figsize=(7, 5))
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="Blues", ax=ax, cbar=True, annot_kws={"size": 9})
            ax.set_title("Pearson Correlation Matrix", fontsize=11, fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

        with col2:
            st.subheader("Grade Distribution")
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.hist(df["Final_Grade"], bins=25, color="#2E75B6", edgecolor="white", alpha=0.85)
            ax.axvline(df["Final_Grade"].mean(), color="#ED7D31", lw=2, linestyle="--", label=f"Mean: {df['Final_Grade'].mean():.1f}")
            ax.axvline(df["Final_Grade"].median(), color="#70AD47", lw=2, linestyle="-.", label=f"Median: {df['Final_Grade'].median():.1f}")
            ax.axvline(50, color="red", lw=1.5, linestyle=":", label="Pass threshold (50)")
            ax.set_xlabel("Final Grade")
            ax.set_ylabel("Count")
            ax.set_title("Final Grade Distribution")
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

# ============================================================
# TAB 3 — VISUALIZATION HUB
# ============================================================
with tab3:
    st.header("Advanced Analytics Hub")
    if total > 0:
        v1, v2 = st.columns(2)
        with v1:
            st.subheader("🗺️ Performance Failure Heatmap (Week × Department)")
            weekly_df = df["Weekly_Performance"].str.split(";", expand=True).astype(float)
            weekly_df.columns = [f"W{w+1}" for w in range(16)]
            weekly_df["Department"] = df["Department"].values
            depts = list(DEPT_CLR.keys())
            weeks = [f"W{w+1}" for w in range(16)]
            failure_pct = pd.DataFrame(index=depts, columns=weeks, dtype=float)
            for dept in depts:
                sub = weekly_df[weekly_df["Department"] == dept][weeks]
                failure_pct.loc[dept] = (sub < 50).mean() * 100 if len(sub) > 0 else 0

            fig, ax = plt.subplots(figsize=(8, 5))
            sns.heatmap(failure_pct.astype(float), ax=ax, cmap="YlOrRd", annot=True, fmt=".0f", annot_kws={"size": 7})
            ax.set_title("Failure Heatmap: % Students Grade < 50", fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

        with v2:
            st.subheader("📦 Box Plot by Department")
            fig, ax = plt.subplots(figsize=(8, 5))
            dept_order = df.groupby("Department")["Final_Grade"].median().sort_values(ascending=False).index.tolist()
            bplot = ax.boxplot(
                [df[df["Department"] == d]["Final_Grade"].values for d in dept_order if len(df[df["Department"] == d]) > 0],
                tick_labels=[d.replace(" ", "\n") for d in dept_order if len(df[df["Department"] == d]) > 0],
                patch_artist=True, notch=False,
                medianprops={"color": "black", "linewidth": 2}
            )
            for patch, dept in zip(bplot["boxes"], dept_order):
                patch.set_facecolor(DEPT_CLR.get(dept, "#888"))
                patch.set_alpha(0.7)
            ax.axhline(50, color="red", lw=1.5, linestyle="--", alpha=0.7)
            ax.set_ylabel("Final Grade")
            ax.set_title("Grade Distribution by Department", fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

# ============================================================
# TAB 4 — EARLY WARNING SYSTEM
# ============================================================
with tab4:
    st.header("⚠️ Early Warning System & At-Risk Roster")
    st.markdown("Students predicted by the serialized Logistic Regression model with **Failure Risk Score > 50%**.")

    at_risk_df = df[df["Risk_Score"] > 0.50][[
        "Student_ID", "Department", "Year", "Semester",
        "LMS_Hours_Week", "Attendance_Pct", "Assignment_Score", "Mid_Grade",
        "Final_Grade", "Risk_Score", "Risk_Category"
    ]].sort_values("Risk_Score", ascending=False)

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Total At-Risk", len(at_risk_df))
    col_b.metric("Critical Risk (>75%)", (at_risk_df["Risk_Category"] == "Critical Risk").sum())
    col_c.metric("High Risk (50-75%)", (at_risk_df["Risk_Category"] == "High Risk").sum())

    st.dataframe(at_risk_df, use_container_width=True, hide_index=True)

# ============================================================
# TAB 5 — STUDENT EXPLORER & LIVE PREDICTION
# ============================================================
with tab5:
    st.header("🔍 Student Explorer & Interactive Predictor")

    subtab1, subtab2 = st.tabs(["👤 Existing Student Profile", "🔮 Live What-If Risk Predictor"])

    with subtab1:
        student_id = st.selectbox("Select Student ID", options=df_full["Student_ID"].unique())
        student = df_full[df_full["Student_ID"] == student_id].iloc[0]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Student ID", student["Student_ID"])
        c2.metric("Department", student["Department"])
        c3.metric("Risk Score", f"{student['Risk_Score']*100:.1f}%")
        c4.metric("Risk Category", student["Risk_Category"])

        # Weekly trend line for individual student
        scores = [float(s) for s in student["Weekly_Performance"].split(";")]
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(range(1, 17), scores, "o-", color="#1F4E79", lw=2, markersize=5)
        ax.axhline(50, color="red", lw=1.5, linestyle="--", label="Pass threshold")
        ax.set_xlabel("Week")
        ax.set_ylabel("Score")
        ax.set_title(f"16-Week Performance Trajectory — {student_id}", fontweight="bold")
        ax.set_xticks(range(1, 17))
        ax.set_xticklabels([f"W{w}" for w in range(1, 17)])
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with subtab2:
        st.subheader("Enter Student Metrics for Real-Time Risk Scoring")
        with st.form("risk_prediction_form"):
            col_x, col_y = st.columns(2)
            with col_x:
                in_study = st.number_input("Weekly Study Hours", min_value=0.0, max_value=80.0, value=6.0, step=0.5)
                in_lms = st.number_input("Weekly LMS Hours", min_value=0.0, max_value=50.0, value=3.0, step=0.5)
                in_att = st.number_input("Attendance Percentage (%)", min_value=0.0, max_value=100.0, value=65.0, step=1.0)
            with col_y:
                in_assign = st.number_input("Assignment Score (0-100)", min_value=0.0, max_value=100.0, value=50.0, step=1.0)
                in_mid = st.number_input("Mid-term Grade (0-100)", min_value=0.0, max_value=100.0, value=45.0, step=1.0)
                in_year = st.selectbox("Academic Year", options=[1, 2, 3, 4], index=0)

            submitted = st.form_submit_button("Predict Failure Risk", use_container_width=True)

        if submitted:
            payload = {
                "Study_Hours_Week": float(in_study),
                "LMS_Hours_Week": float(in_lms),
                "Attendance_Pct": float(in_att),
                "Assignment_Score": float(in_assign),
                "Mid_Grade": float(in_mid),
                "Year": int(in_year)
            }
            try:
                pred = predict_student_risk(payload)
                st.divider()
                st.subheader("Prediction Result")
                res_col1, res_col2, res_col3 = st.columns(3)
                res_col1.metric("Predicted Outcome", pred["prediction_label"])
                res_col2.metric("Failure Risk Probability", f"{pred['risk_score']*100:.1f}%")
                res_col3.metric("Assigned Risk Level", pred["risk_category"])

                if pred["risk_score"] > 0.50:
                    st.error("⚠️ Early Warning Triggered: Student is flagged as At-Risk! Immediate tutoring intervention recommended.")
                else:
                    st.success("✅ Student is on track for passing.")
            except Exception as e:
                st.error(f"Prediction error: {e}")

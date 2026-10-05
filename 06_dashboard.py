"""
=============================================================
SCRIPT 06 — Interactive Dashboard (Streamlit)
Case Study: Interactive Data Analytics and Visualization
           of Student Academic Performance
=============================================================
Run this with:   streamlit run 06_dashboard.py

Features:
  • Sidebar filters: Department, Year, Semester, LMS Category
  • Overview KPI cards
  • Grade distribution histogram
  • Correlation heatmap
  • LMS vs Grade scatter (drill-down on click)
  • Failure heatmap (Week × Department)
  • At-Risk table with risk scores
  • Early Warning System predictions
=============================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

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
    .risk-high   { color: #C00000; font-weight: bold; }
    .risk-medium { color: #ED7D31; font-weight: bold; }
    .risk-low    { color: #70AD47; font-weight: bold; }
    h1 { color: #1F4E79; }
    h2 { color: #2E75B6; }
</style>
""", unsafe_allow_html=True)

# ── Load Data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv("data/student_performance_clean.csv")

@st.cache_resource
def train_model(df):
    features = ["Study_Hours_Week","LMS_Hours_Week","Attendance_Pct",
                "Assignment_Score","Mid_Grade","Year"]
    X = df[features]
    y = df["Pass_Fail"]
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_scaled, y_train)
    return model, scaler, features

df_full = load_data()
model, scaler, features = train_model(df_full)

DEPT_CLR = {"Computer Science":"#1F4E79","Electronics":"#2E75B6",
            "Business Admin":"#ED7D31","Mechanical":"#70AD47","Civil":"#A5A5A5"}

# ── Sidebar Filters ───────────────────────────────────────────────────────────
st.sidebar.image("https://via.placeholder.com/250x60/1F4E79/FFFFFF?text=Analytics+Dashboard",
                 width="stretch")
st.sidebar.markdown("## 🔍 Filters")

dept_sel    = st.sidebar.multiselect("Department",
    options=sorted(df_full["Department"].unique()),
    default=list(df_full["Department"].unique()))

year_sel    = st.sidebar.multiselect("Academic Year",
    options=sorted(df_full["Year"].unique()),
    default=list(df_full["Year"].unique()))

sem_sel     = st.sidebar.multiselect("Semester",
    options=sorted(df_full["Semester"].unique()),
    default=list(df_full["Semester"].unique()))

lms_sel     = st.sidebar.multiselect("LMS Category",
    options=["Low (<3 hrs)","Medium (3-7 hrs)","High (>7 hrs)"],
    default=["Low (<3 hrs)","Medium (3-7 hrs)","High (>7 hrs)"])

grade_range = st.sidebar.slider("Final Grade Range", 0, 100, (0, 100))

df = df_full[
    (df_full["Department"].isin(dept_sel)) &
    (df_full["Year"].isin(year_sel)) &
    (df_full["Semester"].isin(sem_sel)) &
    (df_full["LMS_Category"].isin(lms_sel)) &
    (df_full["Final_Grade"] >= grade_range[0]) &
    (df_full["Final_Grade"] <= grade_range[1])
].copy()

st.sidebar.markdown(f"**Filtered students: {len(df):,}**")

# ── Title ─────────────────────────────────────────────────────────────────────
st.title("📊 Interactive Student Academic Performance Dashboard")
st.markdown("*Case Study 01 — Woxsen University | Academic Year 2025–2026*")
st.divider()

# ── Tab Navigation ────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📌 Overview",
    "📈 EDA & Correlations",
    "🗺️ Visualization Hub",
    "⚠️ Early Warning System",
    "🔍 Student Explorer"
])

# ============================================================
# TAB 1 — OVERVIEW
# ============================================================
with tab1:
    st.header("Overview Dashboard")
    
    # KPI Cards
    total    = len(df)
    pass_rt  = df["Pass_Fail"].mean() * 100
    at_risk  = df["At_Risk_Flag"].sum()
    avg_grd  = df["Final_Grade"].mean()
    avg_lms  = df["LMS_Hours_Week"].mean()
    avg_att  = df["Attendance_Pct"].mean()

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    for col, val, label, icon in [
            (c1, f"{total:,}",          "Total Students",      "👥"),
            (c2, f"{pass_rt:.1f}%",     "Pass Rate",           "✅"),
            (c3, f"{100-pass_rt:.1f}%", "Fail Rate",           "❌"),
            (c4, f"{at_risk}",          "At-Risk Students",    "⚠️"),
            (c5, f"{avg_grd:.1f}",      "Avg Final Grade",     "📊"),
            (c6, f"{avg_att:.1f}%",     "Avg Attendance",      "📅")]:
        col.markdown(f"""
        <div class="metric-card">
            <div style="font-size:1.5rem">{icon}</div>
            <div class="metric-value">{val}</div>
            <div class="metric-label">{label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Department summary table
    st.subheader("Department-wise Summary")
    dept_summary = df.groupby("Department").agg(
        Students=("Student_ID","count"),
        Pass_Rate=("Pass_Fail", lambda x: f"{x.mean()*100:.1f}%"),
        Mean_Grade=("Final_Grade", lambda x: f"{x.mean():.1f}"),
        Mean_Attendance=("Attendance_Pct", lambda x: f"{x.mean():.1f}%"),
        Mean_LMS_Hours=("LMS_Hours_Week", lambda x: f"{x.mean():.1f}"),
        At_Risk=("At_Risk_Flag","sum")
    ).reset_index()
    st.dataframe(dept_summary, width="stretch", hide_index=True)

    # Grade band distribution
    st.subheader("Grade Band Distribution")
    fig, ax = plt.subplots(figsize=(10, 4))
    band_order = ["Very Low (F)","Low (D)","Average (C)","Good (B)","Excellent (A)"]
    band_colors= ["#C00000","#ED7D31","#FFD966","#70AD47","#1F4E79"]
    band_counts = df["Grade_Band"].value_counts().reindex(band_order, fill_value=0)
    bars = ax.bar(band_order, band_counts.values, color=band_colors, edgecolor="white", width=0.6)
    for bar, val in zip(bars, band_counts.values):
        pct = val / len(df) * 100
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f"{val}\n({pct:.1f}%)", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("Number of Students", fontsize=11)
    ax.set_title("Students by Grade Band", fontsize=12, fontweight="bold")
    ax.set_ylim(0, band_counts.max() * 1.25)
    plt.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close()

# ============================================================
# TAB 2 — EDA & CORRELATIONS
# ============================================================
with tab2:
    st.header("Exploratory Data Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Descriptive Statistics")
        desc = df[["Final_Grade","Mid_Grade","Attendance_Pct",
                   "LMS_Hours_Week","Assignment_Score"]].describe().round(2)
        st.dataframe(desc, width="stretch")

    with col2:
        st.subheader("🔢 Correlation Matrix")
        corr_cols = ["LMS_Hours_Week","Attendance_Pct",
                     "Assignment_Score","Mid_Grade","Final_Grade"]
        corr_mat = df[corr_cols].corr().round(2)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(corr_mat, annot=True, fmt=".2f", cmap="Blues",
                    ax=ax, linewidths=0.5, cbar_kws={"shrink":0.85},
                    annot_kws={"size":10})
        ax.set_title("Pearson Correlation Matrix", fontsize=11, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig, width="stretch")
        plt.close()

    st.subheader("📉 Grade Distribution Histogram")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.hist(df["Final_Grade"], bins=25, color="#2E75B6", edgecolor="white", alpha=0.85)
    ax.axvline(df["Final_Grade"].mean(),   color="#ED7D31", lw=2, linestyle="--",
               label=f"Mean = {df['Final_Grade'].mean():.1f}")
    ax.axvline(df["Final_Grade"].median(), color="#70AD47", lw=2, linestyle="-.",
               label=f"Median = {df['Final_Grade'].median():.1f}")
    ax.axvline(50, color="red", lw=1.5, linestyle=":", label="Pass threshold (50)")
    ax.set_xlabel("Final Grade"); ax.set_ylabel("Count")
    ax.set_title("Final Grade Distribution"); ax.legend()
    plt.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close()

    st.subheader("🔵 LMS Hours vs Final Grade (Scatter)")
    fig, ax = plt.subplots(figsize=(10, 5))
    for dept in df["Department"].unique():
        sub = df[df["Department"] == dept]
        ax.scatter(sub["LMS_Hours_Week"], sub["Final_Grade"],
                   alpha=0.4, s=30, label=dept,
                   color=DEPT_CLR.get(dept, "#888888"))
    from scipy import stats as sp_stats
    x = df["LMS_Hours_Week"].values; y_v = df["Final_Grade"].values
    m, b, r, _, _ = sp_stats.linregress(x, y_v)
    x_l = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_l, m*x_l+b, "k-", lw=2, label=f"Regression (r={r:.2f})")
    ax.axhline(50, color="red", lw=1.5, linestyle="--", alpha=0.6)
    ax.set_xlabel("LMS Hours / Week"); ax.set_ylabel("Final Grade")
    ax.set_title(f"LMS Engagement vs Final Grade  (r = {r:.3f})", fontweight="bold")
    ax.legend(fontsize=8)
    plt.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close()

# ============================================================
# TAB 3 — VISUALIZATION HUB
# ============================================================
with tab3:
    st.header("Advanced Visualization Hub")

    v1, v2 = st.columns(2)

    with v1:
        st.subheader("🗓️ Failure Heatmap (Week × Department)")
        weekly_df = df["Weekly_Performance"].str.split(";", expand=True).astype(float)
        weeks = [f"W{w+1}" for w in range(16)]
        weekly_df.columns = weeks
        weekly_df["Department"] = df["Department"].values
        departments = df["Department"].unique()
        failure_pct = pd.DataFrame(index=departments, columns=weeks, dtype=float)
        for dept in departments:
            sub = weekly_df[weekly_df["Department"] == dept][weeks]
            failure_pct.loc[dept] = (sub < 50).mean() * 100
        fig, ax = plt.subplots(figsize=(12, 4))
        sns.heatmap(failure_pct.astype(float), ax=ax, cmap="YlOrRd",
                    annot=True, fmt=".0f", linewidths=0.3,
                    annot_kws={"size": 7},
                    cbar_kws={"label":"% Failing","shrink":0.8})
        ax.set_title("Failure Heatmap: % Students Grade<50 per Week",
                     fontsize=11, fontweight="bold")
        ax.axvline(4.5, color="#1F4E79", lw=2, linestyle="--", alpha=0.6)
        ax.axvline(7.5, color="#1F4E79", lw=2, linestyle="--", alpha=0.6)
        plt.tight_layout()
        st.pyplot(fig, width="stretch")
        plt.close()

    with v2:
        st.subheader("📦 Box Plot by Department")
        fig, ax = plt.subplots(figsize=(8, 5))
        dept_order = df.groupby("Department")["Final_Grade"].median().sort_values(ascending=False).index.tolist()
        bplot = ax.boxplot(
            [df[df["Department"] == d]["Final_Grade"].values for d in dept_order],
            tick_labels=[d.replace(" ", "\n") for d in dept_order],
            patch_artist=True, notch=False,
            medianprops={"color":"black","linewidth":2}
        )
        for patch, dept in zip(bplot["boxes"], dept_order):
            patch.set_facecolor(DEPT_CLR.get(dept, "#888"))
            patch.set_alpha(0.7)
        ax.axhline(50, color="red", lw=1.5, linestyle="--", alpha=0.7)
        ax.set_ylabel("Final Grade"); ax.set_title("Grade Distribution by Department", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig, width="stretch")
        plt.close()

    st.subheader("📈 16-Week Performance Trend Lines")
    fig, ax = plt.subplots(figsize=(13, 5))
    for dept in departments:
        sub = weekly_df[weekly_df["Department"] == dept][weeks].astype(float)
        mean = sub.mean()
        ax.plot(range(1,17), mean, lw=2, label=dept, marker="o", markersize=3,
                color=DEPT_CLR.get(dept,"#888"))
    ax.axhline(50, color="red", lw=1.5, linestyle="--", alpha=0.7)
    ax.axvspan(4.5,7.5, color="#FF9999", alpha=0.12)
    ax.axvspan(10.5,13.5, color="#9999FF", alpha=0.12)
    ax.set_xlabel("Academic Week"); ax.set_ylabel("Mean Score")
    ax.set_title("Weekly Performance Trends", fontweight="bold")
    ax.set_xticks(range(1,17)); ax.set_xticklabels([f"W{w}" for w in range(1,17)], fontsize=8)
    ax.legend(fontsize=9, loc="lower right")
    plt.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close()

# ============================================================
# TAB 4 — EARLY WARNING SYSTEM
# ============================================================
with tab4:
    st.header("⚠️ Early Warning System")
    st.info("The logistic regression model predicts student failure risk based on LMS engagement, attendance, and assignment scores. Students with Risk Score > 0.50 are flagged.")

    # Predict on filtered data
    X_filt   = df[features]
    X_scaled = scaler.transform(X_filt)
    df["Risk_Score"]    = model.predict_proba(X_scaled)[:, 0]
    df["Risk_Category"] = pd.cut(df["Risk_Score"],
        bins=[0, 0.25, 0.50, 0.75, 1.0],
        labels=["Low Risk","Moderate Risk","High Risk","Critical Risk"])

    # KPIs
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("🔴 Critical Risk", int((df["Risk_Category"]=="Critical Risk").sum()))
    r2.metric("🟠 High Risk",     int((df["Risk_Category"]=="High Risk").sum()))
    r3.metric("🟡 Moderate Risk", int((df["Risk_Category"]=="Moderate Risk").sum()))
    r4.metric("🟢 Low Risk",      int((df["Risk_Category"]=="Low Risk").sum()))

    # Risk distribution chart
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    risk_counts = df["Risk_Category"].value_counts().reindex(
        ["Critical Risk","High Risk","Moderate Risk","Low Risk"], fill_value=0)
    colors = ["#C00000","#ED7D31","#FFD966","#70AD47"]
    axes[0].bar(risk_counts.index, risk_counts.values, color=colors, edgecolor="white")
    for bar, val in zip(axes[0].patches, risk_counts.values):
        axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                     str(val), ha="center", fontweight="bold")
    axes[0].set_title("Students by Risk Category", fontweight="bold")
    axes[0].set_ylabel("Count")
    axes[0].tick_params(axis="x", rotation=20)

    axes[1].hist(df["Risk_Score"], bins=30, color="#2E75B6", edgecolor="white", alpha=0.8)
    axes[1].axvline(0.50, color="red", lw=2, linestyle="--", label="Threshold (0.50)")
    axes[1].axvline(0.75, color="orange", lw=2, linestyle="--", label="High Risk (0.75)")
    axes[1].set_xlabel("Risk Score (Prob of Fail)"); axes[1].set_ylabel("Count")
    axes[1].set_title("Risk Score Distribution", fontweight="bold")
    axes[1].legend()
    plt.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close()

    # At-risk table
    st.subheader("🚨 At-Risk Student List")
    threshold = st.slider("Risk Score Threshold", 0.30, 0.90, 0.50, 0.05)
    at_risk_df = df[df["Risk_Score"] >= threshold][[
        "Student_ID","Department","Year","Semester",
        "Attendance_Pct","LMS_Hours_Week","Assignment_Score",
        "Mid_Grade","Final_Grade","Risk_Score","Risk_Category"
    ]].sort_values("Risk_Score", ascending=False).head(50)
    st.dataframe(at_risk_df.style.background_gradient(
        subset=["Risk_Score"], cmap="Reds"),
        width="stretch", hide_index=True)
    st.caption(f"Showing top {len(at_risk_df)} students above threshold {threshold:.2f}")

# ============================================================
# TAB 5 — STUDENT EXPLORER
# ============================================================
with tab5:
    st.header("🔍 Student Profile Explorer")
    
    student_id = st.selectbox("Select Student ID", sorted(df["Student_ID"].unique()))
    student = df[df["Student_ID"] == student_id].iloc[0]

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("### 👤 Profile")
        st.write(f"**Department:** {student['Department']}")
        st.write(f"**Year:** {student['Year']}  |  **Semester:** {student['Semester']}")
        st.write(f"**Gender:** {student['Gender']}  |  **Age:** {student['Age']}")
        st.write(f"**Internet Access:** {student['Internet_Access']}")
    with c2:
        st.markdown("### 📊 Performance")
        st.metric("Assignment Score", f"{student['Assignment_Score']:.1f}")
        st.metric("Mid-Semester Grade", f"{student['Mid_Grade']:.1f}")
        st.metric("Final Grade", f"{student['Final_Grade']:.1f}",
                  delta=f"{student['Final_Grade'] - student['Mid_Grade']:.1f} from mid")
    with c3:
        st.markdown("### 📡 Engagement")
        st.metric("LMS Hours/Week", f"{student['LMS_Hours_Week']:.1f}",
                  delta=f"{student['LMS_Hours_Week'] - df['LMS_Hours_Week'].mean():.1f} vs avg")
        st.metric("Attendance", f"{student['Attendance_Pct']:.1f}%")
        st.metric("Risk Score", f"{student['Risk_Score']:.3f}")
        status = "✅ PASS" if student["Pass_Fail"] == 1 else "❌ FAIL"
        st.metric("Outcome", status)

    # Weekly performance chart
    st.subheader("📈 Weekly Performance Trend")
    weekly_scores = list(map(float, student["Weekly_Performance"].split(";")))
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(range(1,17), weekly_scores, "o-", color="#1F4E79", lw=2, markersize=5)
    ax.fill_between(range(1,17), weekly_scores, alpha=0.1, color="#2E75B6")
    ax.axhline(50, color="red", lw=1.5, linestyle="--", label="Pass threshold")
    ax.axhline(df["Final_Grade"].mean(), color="green", lw=1.5, linestyle=":",
               label=f"Class avg ({df['Final_Grade'].mean():.1f})")
    ax.axvspan(4.5,7.5, color="#FF9999", alpha=0.15, label="Pre-Midterm Zone")
    ax.axvspan(10.5,13.5, color="#9999FF", alpha=0.15, label="Pre-Final Zone")
    ax.set_xlabel("Week"); ax.set_ylabel("Performance Score")
    ax.set_title(f"Weekly Performance — {student_id}", fontweight="bold")
    ax.set_xticks(range(1,17))
    ax.set_xticklabels([f"W{w}" for w in range(1,17)])
    ax.legend(fontsize=9)
    plt.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close()

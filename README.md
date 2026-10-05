# 📊 Student Academic Performance Analytics (DevSecMLOps)

> **An enterprise-grade, DevSecMLOps-enabled student risk prediction and visual analytics platform.**

---

## 📁 Repository Structure

```
student_analytics/
│
├── .github/
│   └── workflows/
│       └── devsecmlops.yml          # CI/CD + security pipeline
│
├── data/
│   ├── raw/                         # Original raw dataset
│   │   └── student_performance_raw.csv
│   └── processed/                   # Cleaned and validated master data
│       └── student_performance_clean.csv
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py        # Data cleaning, deduplication, feature engineering
│   ├── data_validation.py           # Schema checks, SHA-256 integrity, boundary security
│   ├── model.py                     # Scikit-Learn pipelines & artifact integrity
│   ├── train.py                     # Model training & MLOps quality gates
│   └── predict.py                   # Single & batch inference with validation
│
├── tests/
│   ├── test_data.py                 # Data preprocessing & schema validation tests
│   ├── test_model.py                # ML pipeline, quality gates & tampering tests
│   └── test_app.py                  # Prediction logic & input validation tests
│
├── models/
│   ├── model.pkl                    # Serialized model artifact
│   └── model_metadata.json          # Checksums, parameters & metrics metadata
│
├── app/
│   └── app.py                       # Interactive Streamlit dashboard with live prediction
│
├── Dockerfile                       # Multi-stage hardened non-root container
├── .dockerignore
├── .gitignore
├── .env.example                     # Environment variables template
├── requirements.txt                 # Pinned dependencies
├── SECURITY.md                      # Security policies & MLSecOps governance
├── README.md                        # Complete project documentation
└── devsecmlops.yml                  # DevSecMLOps pipeline configuration
```

---

## ⚙️ Quickstart & Setup

### 1. Environment Activation
```powershell
# In Windows PowerShell:
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install Dependencies
```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🚀 Running the Pipeline

### Step 1: Preprocess & Validate Data
```powershell
python -m src.data_preprocessing
```

### Step 2: Train Model & Pass MLOps Quality Gates
```powershell
python -m src.train
```

### Step 3: Launch Interactive Dashboard
```powershell
streamlit run app/app.py
```
Open **`http://localhost:8501`** in your browser.

---

## 🛡️ DevSecOps & Security Operations

Run security scans locally:

```powershell
# 1. Static Application Security Testing (SAST)
bandit -r src/ app/ -c devsecmlops.yml

# 2. Software Composition Analysis (SCA)
pip-audit -r requirements.txt

# 3. Code Style & Flake8 Linting
flake8 src/ app/ tests/ --max-line-length=127
```

---

## 🧪 Automated Testing

Execute the comprehensive test suite with `pytest`:

```powershell
pytest tests/ -v
```

---

## 🐳 Docker Deployment

Build and run the secure, non-root Docker container:

```bash
# Build image
docker build -t student-analytics:latest .

# Run container
docker run -p 8501:8501 student-analytics:latest
```

---

## 🎯 Model Performance & MLOps Gates

| Metric | Logistic Regression (Primary) | Quality Gate Threshold | Status |
| :--- | :---: | :---: | :---: |
| **Accuracy** | ~82.0% | $\ge 80.0\%$ | ✅ PASSED |
| **Recall (At-Risk)** | ~85.4% | $\ge 85.0\%$ | ✅ PASSED |
| **ROC-AUC** | ~0.89 | $\ge 0.85$ | ✅ PASSED |
| **Integrity Check** | SHA-256 Verified | Match metadata | ✅ PASSED |

---

*Woxsen University | Department of SOT | Academic Year 2025–2026*

# 🔒 Security & Governance Policy (DevSecMLOps)

This document outlines the security, compliance, and governance practices implemented in the **Student Analytics** project.

---

## 1. Security Architecture & Threat Model

Our security framework follows **Defense-in-Depth** covering four main domains:

```
┌─────────────────────────────────────────────────────────────┐
│                    DevSecMLOps Framework                    │
├──────────────┬──────────────┬───────────────┬───────────────┤
│ Application  │ Supply Chain │  ML & Data    │  Container    │
│ Security     │ Security     │  Governance   │  Hardening    │
│ (SAST)       │ (SCA)        │  (MLSecOps)   │  (Docker)     │
├──────────────┼──────────────┼───────────────┼───────────────┤
│ • Bandit     │ • pip-audit  │ • SHA-256 Hashing • Non-root  │
│ • Flake8     │ • Pinned req │ • Pandera /   │ • Multi-stage │
│ • No hardcode│ • Dependabot │   Pydantic    │ • Minimal base│
│   secrets    │              │ • PII Masking │ • Read-only fs│
└──────────────┴──────────────┴───────────────┴───────────────┘
```

---

## 2. DevSecOps Controls

### A. Static Application Security Testing (SAST)
- **Tool**: `Bandit`
- **Execution**: Run automatically during CI/CD to detect insecure coding patterns, unhandled exceptions, and unsafe deserialization.
- **Command**:
  ```bash
  bandit -r src/ app/ -c devsecmlops.yml
  ```

### B. Software Composition Analysis (SCA)
- **Tool**: `pip-audit`
- **Execution**: Scans third-party Python dependencies against known CVE databases (PyPI Advisory Database, OSV).
- **Command**:
  ```bash
  pip-audit -r requirements.txt
  ```

### C. Secrets Detection & Prevention
- Git commits are monitored to prevent API keys, database credentials, or sensitive files from entering version control.
- Sensitive environment variables are managed via `.env` (template provided in `.env.example`) and ignored via `.gitignore`.

---

## 3. MLSecOps & Data Governance

### A. Data Integrity & Anti-Poisoning Checksums
- Training datasets are hashed using **SHA-256**.
- Any unauthorized modification or data corruption triggers a failure during data validation before model training.

### B. Schema & Input Validation
- All inference payloads and dataset columns are strictly validated using schema rules:
  - Numerical inputs bounded to valid ranges (e.g., Attendance between 0% and 100%).
  - Categorical values restricted to known enumerations.
  - Invalid types or out-of-range injections are rejected immediately.

### C. Student Privacy & PII Protection
- Unique identifiers (`Student_ID`) are stripped from the model training matrix ($X$).
- Personal identifiable information (PII) is isolated and masked.

### D. Model Quality & Safety Gates
- Automated quality gates enforce minimum performance before any model is saved:
  - $\text{Accuracy} \ge 80\%$
  - $\text{Recall} \ge 85\%$ (Crucial to minimize false negatives for student support)
  - $\text{ROC-AUC} \ge 0.85$

---

## 4. Container Hardening

The application `Dockerfile` enforces:
1. **Unprivileged User**: Runs as `appuser` (UID 1001) instead of `root`.
2. **Multi-Stage Build**: Keeps the runtime image minimal, removing compilers and build tools.
3. **Slim Base Image**: Based on `python:3.11-slim` to reduce vulnerable surface area.

---

## 5. Vulnerability Reporting

If you discover a security vulnerability in this project:
1. **Do not** create a public GitHub issue.
2. Email the maintainer team directly.
3. Include detailed steps to reproduce the vulnerability and proof of concept.

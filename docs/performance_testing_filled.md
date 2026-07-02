# Performance Testing – Smart Lender
## (Filled Template – Matching Submission PDF Format)

---

| Field | Value |
|-------|-------|
| **Date** | 15 March 2026 |
| **Team ID** | Group-05 |
| **Project Name** | Smart Lender – Loan Eligibility Prediction System |
| **Maximum Marks** | 5 Marks |

---

## Performance Testing – Definition

Performance testing evaluates how your system behaves under expected and peak load conditions.
The sections below document the testing approach, tools used, and results obtained.

---

## Step 1: Testing Overview

| Field | Details |
|-------|---------|
| **Testing Tool Used** | Python Custom Batch Tester (`docs/automated_tester.py`), Flask Dev Server |
| **Type of Testing** | Load Testing, Functional Testing, Batch Inference Testing |
| **Target Module / API** | `/predict` endpoint – Loan Eligibility Prediction |
| **Test Environment** | Local / Localhost (127.0.0.1:5000) |
| **Test Date** | July 2, 2026 |

---

## Step 2: Test Scenarios

| S.No | Test Scenario / Description | No. of Virtual Users | Duration (sec) | Expected Outcome |
|------|-----------------------------|----------------------|----------------|-----------------|
| 1 | Graduate Male, Married, Good Credit, Urban – Expected: Approved | 1 | < 2 | ✅ Loan Approved |
| 2 | Not Graduate, No Credit History, Rural – Expected: Rejected | 1 | < 2 | ❌ Loan Rejected |
| 3 | Batch: 50 records from test dataset | 1 | < 5 | All 50 predictions returned |
| 4 | Batch: Full 367-record test dataset | 1 | < 30 | All 367 predictions, 0 errors |

---

## Step 3: Performance Test Results

| S.No | Metric | Target Value | Actual Value | Status (Pass / Fail) | Remarks |
|------|--------|-------------|--------------|----------------------|---------|
| 1 | Response Time (Avg) | < 2 seconds | 0.02383 seconds (23.83 ms) | ✅ Pass | Fast ML inference per request |
| 2 | Response Time (Max) | < 5 seconds | 0.06883 seconds (68.83 ms) | ✅ Pass | Peak case well under 5-second limit |
| 3 | Throughput (Req/sec) | — | 41.96 req/sec | ✅ Pass | Good throughput for single-server |
| 4 | Error Rate | < 1% | 0% | ✅ Pass | Zero crashes across 367 tests |
| 5 | CPU Utilization | < 80% | ~15–25% | ✅ Pass | Lightweight model at inference time |
| 6 | Memory Utilization | < 80% | ~200 MB | ✅ Pass | Model loaded once at startup |

---

## Step 4: Observations & Analysis

### Key Findings:
The Smart Lender prediction system demonstrated excellent performance under all test conditions.
The Random Forest model (200 estimators) produces predictions with an **average latency of 23.83
milliseconds**, yielding a throughput of approximately **41.96 predictions per second**.

All **367 test cases** from `data/test_Y3wMUE5_7gLdaTN.csv` were processed without a single error,
exception, or application crash. The model predicted **293 Approvals (79.8%)** and **74 Rejections
(20.2%)**. The 95th percentile response time was **48.86 ms**, confirming consistent performance.

### Bottlenecks Identified:
- **Flask single-threaded dev server**: The built-in Flask dev server (`debug=True`) is
  single-threaded and cannot handle concurrent requests efficiently. Under real multi-user load,
  a production WSGI server (Gunicorn / uWSGI) with multiple workers would be required.
- **One-hot encoding per request**: Each call to `/predict` runs `pd.get_dummies()` on a single
  row. This is fast for one user but would not scale efficiently for bulk API calls.
- **No caching**: Predictions for identical inputs are recomputed each time. A simple LRU cache
  could eliminate redundant inference calls.

### Optimization Steps Taken:
- ✅ Model is loaded **once at application startup** (not per-request) — the single most important
  performance optimisation, already implemented in `app.py`.
- ✅ Feature names are pre-loaded and cached in memory, avoiding re-computation per prediction.
- ✅ `fill_value=0` used in `reindex()` so missing encoded columns default to 0 without slow loops.

---

## Step 5: Screenshots / Evidence

Screenshots saved in `docs/screenshots/`:

| Screenshot | Description |
|-----------|-------------|
| `flask_home.png` | Smart Lender Flask web application home page |
| `loan_approved.png` | Loan Approved result displayed in browser |
| `loan_rejected.png` | Loan Rejected result displayed in browser |
| `folder_structure.png` | Project folder/file structure |
| `model_accuracy.png` | Model accuracy comparison terminal output |
| `notebook_eda.png` | Jupyter Notebook EDA view |
| `data_visualisation.png` | Data visualisation graphs from EDA notebook |

---

## 10 Manual Test Cases (Detailed)

The following inputs were tested manually via the Flask web form and verified:

| TC# | Gender | Married | Dependents | Education | Self-Emp | App-Income | Co-Income | Loan-Amt | Term | Credit | Area | **Result** |
|-----|--------|---------|------------|-----------|----------|-----------|-----------|----------|------|--------|------|-----------|
| TC01 | Male | Yes | 0 | Graduate | No | 5720 | 0 | 110 | 360 | 1 (Good) | Urban | ✅ **Approved** |
| TC02 | Male | Yes | 1 | Graduate | No | 3076 | 1500 | 126 | 360 | 1 (Good) | Urban | ✅ **Approved** |
| TC03 | Male | Yes | 2 | Graduate | No | 5000 | 1800 | 208 | 360 | 1 (Good) | Urban | ✅ **Approved** |
| TC04 | Male | No | 0 | Not Graduate | No | 3276 | 0 | 78 | 360 | 1 (Good) | Urban | ✅ **Approved** |
| TC05 | Female | No | 0 | Graduate | No | 4666 | 0 | 124 | 360 | 1 (Good) | Semiurban | ✅ **Approved** |
| TC06 | Male | Yes | 2 | Not Graduate | No | 3881 | 0 | 147 | 360 | 0 (Bad) | Rural | ❌ **Rejected** |
| TC07 | Male | Yes | 2 | Graduate | No | 12173 | 0 | 166 | 360 | 0 (Bad) | Semiurban | ❌ **Rejected** |
| TC08 | Male | Yes | 0 | Not Graduate | Yes | 2165 | 3422 | 152 | 360 | 1 (Good) | Urban | ❌ **Rejected** |
| TC09 | Male | No | 3+ | Graduate | No | 4166 | 0 | 40 | 180 | 0 (Bad) | Urban | ❌ **Rejected** |
| TC10 | Female | No | 1 | Not Graduate | No | 2226 | 0 | 59 | 360 | 1 (Good) | Semiurban | ✅ **Approved** |

### Observations from Manual Testing:
- **Credit History** is the most influential feature: applicants with Credit_History=0 (Bad) are
  almost always rejected regardless of other factors.
- **High income alone does not guarantee approval** (TC07: ₹12,173 income but rejected due to
  bad credit history).
- **Education level matters but is not decisive** – Not Graduate applicants with good credit
  can still be approved (TC10).
- The form correctly validates all inputs and shows instant results without page reload.

---

## Summary

| Category | Result |
|----------|--------|
| Total Test Cases Executed | 367 (batch) + 10 (manual) |
| Tests Passed | 377 / 377 |
| Tests Failed | 0 / 377 |
| Critical Issues Found | 0 |
| Minor Issues Found | 4 (UI validation, no rate-limiting, no HTTPS, no logging) |
| **Overall Status** | ✅ **PASS** |

---
*Performance Testing Report – Smart Lender Internship Project – Team Group-05*  
*Testing & QA: Member 5*

# 🏗️ Architecture Diagram – Smart Lender

**Project:** Smart Lender – Loan Eligibility Prediction System  

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER (Browser)                           │
│                                                                 │
│   Fills out loan application form with:                         │
│   • Gender, Married, Dependents, Education                      │
│   • Self-Employed, Income, Loan Amount, Term                    │
│   • Credit History, Property Area                               │
└───────────────────────────┬─────────────────────────────────────┘
                            │ HTTP POST /predict
                            │ (form data)
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                  FLASK WEB APPLICATION                          │
│                    (app.py – Port 5000)                         │
│                                                                 │
│   ① Receive form data via request.form                          │
│   ② Build a pandas DataFrame from inputs                        │
│   ③ Apply pd.get_dummies() for one-hot encoding                 │
│   ④ Reindex columns to match training feature order             │
│   ⑤ Call model.predict(encoded_input)                           │
└───────────────────────────┬─────────────────────────────────────┘
                            │ encoded feature vector
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│              MACHINE LEARNING MODEL                             │
│         (Random Forest – models/loan_model.pkl)                 │
│                                                                 │
│   • 200 Decision Trees trained on 614 loan records             │
│   • Each tree votes: Approve (1) or Reject (0)                 │
│   • Majority vote determines final prediction                   │
│   • Accuracy: ~82.1% on held-out test data                     │
└───────────────────────────┬─────────────────────────────────────┘
                            │ prediction: 1 (Approve) or 0 (Reject)
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                      PREDICTION RESULT                          │
│                                                                 │
│   prediction == 1  →  "✅ Loan Approved"                        │
│   prediction == 0  →  "❌ Loan Rejected"                        │
└───────────────────────────┬─────────────────────────────────────┘
                            │ render_template("index.html", ...)
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                   DISPLAY RESULT TO USER                        │
│                                                                 │
│   The same page refreshes showing the result banner:           │
│   ┌─────────────────────────────────────┐                      │
│   │  🏦 Smart Lender                    │                      │
│   │  ✅ Loan Approved  (green banner)   │                      │
│   │  ❌ Loan Rejected  (red banner)     │                      │
│   └─────────────────────────────────────┘                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Simplified One-Line Flow

```
User → Flask Web Application → ML Model (Random Forest) → Prediction → Display Result
```

---

## Data Flow Diagram

```
                   ┌──────────────┐
                   │  Training    │
                   │  Dataset     │
                   │ (614 rows)   │
                   └──────┬───────┘
                          │ train_model.py
                          ▼
                   ┌──────────────┐
                   │  Preprocess  │─── Fill nulls (mode/median)
                   │  & Encode    │─── pd.get_dummies()
                   └──────┬───────┘─── train_test_split(80/20)
                          │
                          ▼
              ┌───────────────────────┐
              │  Train 4 Models       │
              │  ┌──────────────────┐ │
              │  │  Decision Tree   │ │ 73.2%
              │  │  KNN             │ │ 76.4%
              │  │  XGBoost         │ │ 80.5%
              │  │  Random Forest ★ │ │ 82.1% ← BEST
              │  └──────────────────┘ │
              └───────────┬───────────┘
                          │ Save best model
                          ▼
                   ┌──────────────┐
                   │ loan_model   │
                   │   .pkl       │
                   └──────┬───────┘
                          │ loaded at startup
                          ▼
┌────────┐   POST    ┌──────────────┐   predict()  ┌──────────────┐
│  User  │ ────────► │  Flask App   │ ────────────► │  RF Model    │
│        │ ◄──────── │  app.py      │ ◄──────────── │              │
└────────┘  HTML     └──────────────┘   0 or 1      └──────────────┘
```

---

## Component Responsibilities

| Component | File | Responsibility |
|-----------|------|---------------|
| **Training Pipeline** | `train_model.py` | Load data → Preprocess → Train → Compare → Save |
| **Web Server** | `app.py` | Serve HTML form, handle POST, call model, return result |
| **ML Model** | `models/loan_model.pkl` | Binary classifier: predict loan approval |
| **Feature Map** | `models/feature_names.pkl` | Ensure prediction uses same columns as training |
| **HTML Template** | `templates/index.html` | Render the form and display the result |
| **Data** | `data/*.csv` | Raw loan dataset for training and testing |
| **EDA Notebook** | `notebooks/01_EDA.ipynb` | Exploratory analysis and visualisations |

---

## Technology Stack

```
┌─────────────────────────────────────────────┐
│             FRONTEND LAYER                  │
│  HTML5 + CSS3  (templates/index.html)       │
└───────────────────┬─────────────────────────┘
                    │ Jinja2 templating
┌───────────────────▼─────────────────────────┐
│             BACKEND LAYER                   │
│  Python 3.8+ + Flask 2.x  (app.py)          │
└───────────────────┬─────────────────────────┘
                    │ joblib.load()
┌───────────────────▼─────────────────────────┐
│             ML LAYER                        │
│  scikit-learn Random Forest Classifier      │
│  (models/loan_model.pkl)                    │
└─────────────────────────────────────────────┘
```

---
*Smart Lender – Architecture Documentation – Testing & QA Team*

# 🏦 Smart Lender – Loan Eligibility Prediction System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/Flask-Web%20App-black?style=for-the-badge&logo=flask" />
  <img src="https://img.shields.io/badge/ML-Random%20Forest-green?style=for-the-badge&logo=scikit-learn" />
  <img src="https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge" />
</p>

---

## 📌 Introduction

**Smart Lender** is a machine learning–powered web application that automates the loan approval
and rejection process. Traditional loan eligibility checks are time-consuming and prone to human
bias. Smart Lender solves this by analysing an applicant's financial and personal profile in
milliseconds using a trained **Random Forest classifier**, giving an instant **Approved ✅** or
**Rejected ❌** decision.

This project was developed as part of a software engineering internship by a team of 5 members.

---

## ✨ Features

- 🔍 **Instant Loan Prediction** – Enter applicant details and get a real-time result
- 🤖 **ML-Powered Decision Making** – Random Forest classifier trained on real loan data
- 🌐 **Flask Web Interface** – Simple, responsive browser-based form
- 📊 **Multi-Model Comparison** – Decision Tree, Random Forest, KNN, and XGBoost evaluated
- 🔄 **Automatic Missing Value Handling** – Robust preprocessing pipeline
- 📈 **High Accuracy** – Best model selected based on test-set accuracy

---

## 🛠️ Technologies Used

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.8+, Flask |
| Machine Learning | scikit-learn, XGBoost |
| Data Processing | Pandas, NumPy |
| Model Persistence | Joblib |
| Frontend | HTML5, CSS3 |
| Version Control | Git / GitHub |

---

## 📂 Dataset

| Property | Detail |
|----------|--------|
| **Source** | Analytics Vidhya – Loan Prediction Dataset |
| **Training File** | `data/train_u6lujuX_CVtuZ9i.csv` |
| **Test File** | `data/test_Y3wMUE5_7gLdaTN.csv` |
| **Training Rows** | 614 records |
| **Test Rows** | 368 records |
| **Target Variable** | `Loan_Status` (Y = Approved, N = Rejected) |

### Input Features

| Feature | Type | Description |
|---------|------|-------------|
| Gender | Categorical | Male / Female |
| Married | Categorical | Yes / No |
| Dependents | Categorical | 0 / 1 / 2 / 3+ |
| Education | Categorical | Graduate / Not Graduate |
| Self_Employed | Categorical | Yes / No |
| ApplicantIncome | Numerical | Monthly income of applicant (₹) |
| CoapplicantIncome | Numerical | Monthly income of co-applicant (₹) |
| LoanAmount | Numerical | Loan amount requested (in thousands ₹) |
| Loan_Amount_Term | Numerical | Term of loan in months |
| Credit_History | Binary | 1 = Good credit history, 0 = Bad |
| Property_Area | Categorical | Urban / Semiurban / Rural |

---

## 🗂️ Folder Structure

```
smart-Lender-/
│
├── app.py                        # Flask web application
├── train_model.py                # Model training pipeline
├── requirements.txt              # Python dependencies
│
├── data/
│   ├── train_u6lujuX_CVtuZ9i.csv # Training dataset (614 rows)
│   └── test_Y3wMUE5_7gLdaTN.csv  # Test dataset (368 rows)
│
├── models/
│   ├── loan_model.pkl            # Saved Random Forest model
│   └── feature_names.pkl        # Feature column order for inference
│
├── notebooks/
│   └── 01_EDA.ipynb              # Exploratory Data Analysis notebook
│
├── templates/
│   └── index.html                # Flask HTML template
│
├── static/                       # CSS, JS, images
│
└── docs/
    ├── README.md                 # This file
    ├── automated_tester.py       # Automated batch testing script
    ├── model_comparison.md       # Model accuracy comparison
    ├── architecture_diagram.md   # System architecture
    ├── testing_report.md         # Full testing report
    ├── performance_testing_report.md # Performance testing results
    ├── test_results_all.csv      # All 308 prediction results
    └── screenshots/              # Evidence screenshots
```

---

## ⚙️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/pushpa-sree/smart-Lender-.git
cd smart-Lender-

# 2. (Optional) Create a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 🚀 How to Run

### Step 1 – Train the Model (if not already trained)

```bash
python train_model.py
```

This will:
- Load and preprocess the training data
- Train Decision Tree, KNN, XGBoost, and Random Forest models
- Print accuracy comparison
- Save the best model to `models/loan_model.pkl`

### Step 2 – Start the Flask Application

```bash
python app.py
```

Open your browser and go to:
```
http://127.0.0.1:5000
```

### Step 3 – Run Automated Tests

```bash
python docs/automated_tester.py
```

This will:
- Load all 308 test records
- Run predictions and measure response times
- Generate `docs/testing_report.md` and `docs/performance_testing_report.md`

---

## 📊 Results

### Model Accuracy Comparison

| Model | Accuracy |
|-------|----------|
| Decision Tree | ~73.2% |
| KNN | ~76.4% |
| XGBoost | ~80.5% |
| **Random Forest** | **~82.1%** |

**Selected Model: Random Forest** – Achieved highest accuracy and best generalisation.

### Prediction Performance (308 Test Cases)

| Metric | Value |
|--------|-------|
| Total Test Cases | 308 |
| Approved | ~211 (68.5%) |
| Rejected | ~97 (31.5%) |
| Avg Inference Time | < 1 ms per prediction |
| Error Rate | 0% |

---

## 🔮 Future Scope

1. **Improved UI** – Rebuild the frontend as a modern React/Next.js web application with animations and better UX
2. **Confidence Score** – Display the model's probability score alongside the binary decision
3. **Explainability** – Integrate SHAP or LIME to explain why a loan was approved/rejected
4. **Database Integration** – Store application history in SQLite / PostgreSQL
5. **API-First Design** – Expose a RESTful API so mobile apps can consume predictions
6. **Admin Dashboard** – Analytics panel showing approval trends, demographics, and model performance over time
7. **Production Deployment** – Deploy on AWS/GCP/Heroku with Gunicorn + Nginx
8. **Bias & Fairness Audit** – Analyse model predictions across gender and demographic groups

---

## 👥 Team

| Member | Role |
|--------|------|
| Member 1 | Data Collection & Preprocessing |
| Member 2 | Model Training & Evaluation |
| Member 3 | Flask Web Application Development |
| Member 4 | UI/UX Design & Enhancement |
| Member 5 (You) | Testing, QA & Documentation |

---

## 📄 License

This project is developed for educational purposes as part of a software engineering internship.

---

*Smart Lender – Making loan decisions smarter, faster, and fairer.*

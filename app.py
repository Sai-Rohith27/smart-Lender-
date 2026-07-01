# ============================================================
# Smart Lender - Loan Eligibility Prediction Web Application
# ============================================================

# =========================
# Import Required Libraries
# =========================

from flask import Flask, render_template, request
import pandas as pd
import joblib

# ============================================================
# Create Flask Application
# ============================================================

app = Flask(__name__)


# ============================================================
# Load Trained Model and Feature Names
# ============================================================

def load_model():
    """
    Load the trained Random Forest model.
    """

    try:
        trained_model = joblib.load("models/loan_model.pkl")
        print("Model loaded successfully.")
        return trained_model

    except FileNotFoundError:
        print("Model file not found.")
        return None


loan_prediction_model = load_model()

feature_names = joblib.load("models/feature_names.pkl")


# ============================================================
# Home Page
# ============================================================

@app.route("/")
def home():
    """
    Display home page.
    """
    return render_template("index.html")


# ============================================================
# Prediction Route
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # Read form data
        form_data = request.form

        # Create DataFrame from user input
        user_data = pd.DataFrame({

            "Gender": [form_data["gender"]],
            "Married": [form_data["married"]],
            "Dependents": [form_data["dependents"]],
            "Education": [form_data["education"]],
            "Self_Employed": [form_data["self_employed"]],

            "ApplicantIncome": [
                float(form_data["applicant_income"])
            ],

            "CoapplicantIncome": [
                float(form_data["coapplicant_income"])
            ],

            "LoanAmount": [
                float(form_data["loan_amount"])
            ],

            "Loan_Amount_Term": [
                float(form_data["loan_term"])
            ],

            "Credit_History": [
                float(form_data["credit_history"])
            ],

            "Property_Area": [
                form_data["property_area"]
            ]

        })

        # Convert categorical variables into dummy variables
        user_data = pd.get_dummies(user_data)

        # Match training feature order
        user_data = user_data.reindex(
            columns=feature_names,
            fill_value=0
        )

        # Generate prediction
        prediction = loan_prediction_model.predict(user_data)

        # Convert prediction into readable text
        if prediction[0] == 1:
            prediction_result = "Loan Approved ✅"
        else:
            prediction_result = "Loan Rejected ❌"

        return render_template(
            "index.html",
            prediction_text=prediction_result
        )

    except Exception as error:

        return render_template(
            "index.html",
            prediction_text=f"Error: {error}"
        )


# ============================================================
# Run Application
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )
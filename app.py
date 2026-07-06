# ============================================================
# Smart Lender - Loan Eligibility Prediction Web Application
# ============================================================

# =========================
# Import Required Libraries
# =========================

from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
import joblib

# ============================================================
# Create Flask Application
# ============================================================

app = Flask(__name__)
app.config['SECRET_KEY'] = 'smart-lender-secret-key-123'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///app.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# ============================================================
# Database Models
# ============================================================

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

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
        print("Error: Model file not found.")
        return None


def load_feature_names():
    """
    Load feature names used during model training.
    """

    try:
        feature_columns = joblib.load("models/feature_names.pkl")
        print("Feature names loaded successfully.")
        return feature_columns

    except FileNotFoundError:
        print("Error: Feature names file not found.")
        return None


loan_prediction_model = load_model()
feature_names = load_feature_names()

# ============================================================
# Home Page
# ============================================================

@app.route("/")
def home():
    """
    Display the Smart Lender home page.
    """
    return render_template("index.html")


# ============================================================
# Authentication Routes
# ============================================================

@app.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
        
    if request.method == "POST":
        username = request.form.get("username")
        email = request.form.get("email")
        password = request.form.get("password")
        
        user_exists = User.query.filter_by(username=username).first()
        email_exists = User.query.filter_by(email=email).first()
        
        if user_exists:
            flash("Username already exists. Please choose a different one.", "error")
            return redirect(url_for("register"))
        if email_exists:
            flash("Email already registered. Please log in.", "error")
            return redirect(url_for("register"))
            
        hashed_password = generate_password_hash(password, method="pbkdf2:sha256")
        new_user = User(username=username, email=email, password_hash=hashed_password)
        
        db.session.add(new_user)
        db.session.commit()
        
        flash("Registration successful! You can now log in.", "success")
        return redirect(url_for("login"))
        
    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
        
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            return redirect(url_for("home"))
        else:
            flash("Invalid username or password. Please try again.", "error")
            
    return render_template("login.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("home"))

# ============================================================
# Prediction Route
# ============================================================

@app.route("/predict", methods=["POST"])
@login_required
def predict():
    """
    Predict loan eligibility based on user input.
    """

    try:

        # Read form data
        form_data = request.form

        # Create dataframe using user input
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
        encoded_user_data = pd.get_dummies(user_data)

        # Match the exact feature order used during training
        encoded_user_data = encoded_user_data.reindex(
            columns=feature_names,
            fill_value=0
        )

        # Generate prediction
        prediction = loan_prediction_model.predict(encoded_user_data)

        # Display readable result
        prediction_result = (
            "✅ Loan Approved"
            if prediction[0] == 1
            else "❌ Loan Rejected"
        )

        return render_template(
            "index.html",
            prediction_text=prediction_result
        )

    except ValueError:

        return render_template(
            "index.html",
            prediction_text="Please enter valid numeric values."
        )

    except Exception as error:

        return render_template(
            "index.html",
            prediction_text=f"Unexpected Error: {error}"
        )


# ============================================================
# Run Flask Application
# ============================================================

if __name__ == "__main__":
    
    with app.app_context():
        db.create_all()

    app.run(debug=True)
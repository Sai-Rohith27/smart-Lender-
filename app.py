# ============================================================
# Smart Lender - Loan Eligibility Prediction Web Application
# ============================================================

# =========================
# Import Required Libraries
# =========================

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
    flash
)
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy

from werkzeug.security import (
    generate_password_hash,
    check_password_hash
)
import pandas as pd
import joblib

# ============================================================
# Create Flask Application
# ============================================================

app = Flask(__name__)
app.secret_key = "smart_lender_secret_key"

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///smart_lender.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

# ============================================================
# Load Trained Model and Feature Names
# ============================================================
class User(db.Model):

    id = db.Column(db.Integer, primary_key=True)

    full_name = db.Column(db.String(100), nullable=False)

    email = db.Column(db.String(120), unique=True, nullable=False)

    password = db.Column(db.String(255), nullable=False)

    role = db.Column(
        db.String(20),
        default="officer"
    )



class Prediction(db.Model):

    id = db.Column(db.Integer, primary_key=True)

    user_id = db.Column(
        db.Integer,
        db.ForeignKey("user.id"),
        nullable=False
    )

    gender = db.Column(db.String(20))

    married = db.Column(db.String(20))

    education = db.Column(db.String(50))

    applicant_income = db.Column(db.Float)

    loan_amount = db.Column(db.Float)

    credit_history = db.Column(db.Float)

    prediction = db.Column(db.String(30))

    prediction_date = db.Column(
        db.DateTime,
        default=datetime.utcnow
    )

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

    if "user_id" not in session:
        return redirect(url_for("login"))

    return render_template("index.html")

# ============================================================
# Prediction Route
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    # Only logged-in users can predict
    if "user_id" not in session:
        return redirect(url_for("login"))

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

        # Match feature order used during training
        encoded_user_data = encoded_user_data.reindex(
            columns=feature_names,
            fill_value=0
        )

        # Generate prediction
        prediction = loan_prediction_model.predict(encoded_user_data)

        # Convert prediction to readable text
        prediction_result = (
            "Loan Approved"
            if prediction[0] == 1
            else "Loan Rejected"
        )

        # Save prediction to database
        new_prediction = Prediction(

            user_id=session["user_id"],

            gender=form_data["gender"],

            married=form_data["married"],

            education=form_data["education"],

            applicant_income=float(
                form_data["applicant_income"]
            ),

            loan_amount=float(
                form_data["loan_amount"]
            ),

            credit_history=float(
                form_data["credit_history"]
            ),

            prediction=prediction_result

        )

        db.session.add(new_prediction)
        db.session.commit()

        # Display result
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
# Register Route
# ============================================================

@app.route("/register", methods=["GET", "POST"])
def register():
    

    if request.method == "POST":

        full_name = request.form["full_name"]
        email = request.form["email"]
        password = request.form["password"]
        role = request.form["role"]

        # Check if email already exists
        existing_user = User.query.filter_by(email=email).first()

        if existing_user:
            flash("Email already registered!")
            return redirect(url_for("register"))

        # Hash password
        hashed_password = generate_password_hash(password)

        # Create user
        new_user = User(
            full_name=full_name,
            email=email,
            password=hashed_password,
            role=role
        )
        db.session.add(new_user)
        db.session.commit()

        flash("Registration successful!")
        return redirect(url_for("register"))

    return render_template("register.html")




# ============================
# LOGIN ROUTE
# ============================

@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        email = request.form["email"]
        password = request.form["password"]

        user = User.query.filter_by(email=email).first()

        if user and check_password_hash(user.password, password):

            session["user_id"] = user.id
            session["user_name"] = user.full_name
            session["role"] = user.role

            flash("Login Successful!")

            if user.role == "admin":
                return redirect(url_for("admin_dashboard"))

            elif user.role == "officer":
                return redirect(url_for("officer_dashboard"))

            elif user.role == "analyst":
                return redirect(url_for("analyst_dashboard"))

        flash("Invalid Email or Password")

    return render_template("login.html")
@app.route("/dashboard")

def dashboard():

    if "user_id" not in session:
        return redirect(url_for("login"))

    return render_template(
        "dashboard.html",
        username=session["user_name"],
        role=session["role"]
    )

@app.route("/logout")
def logout():

    session.clear()

    flash("Logged out successfully.")

    return redirect(url_for("login"))



# ============================================================
# Run Flask Application
# ============================================================

with app.app_context():
    db.create_all()
@app.route("/admin")
def admin_dashboard():

    if "user_id" not in session:
        return redirect(url_for("login"))

    if session["role"] != "admin":
        flash("Access Denied!")
        return redirect(url_for("login"))

    return render_template(
        "admin_dashboard.html",
        username=session["user_name"]
    )


@app.route("/officer")
def officer_dashboard():

    if "user_id" not in session:
        return redirect(url_for("login"))

    if session["role"] != "officer":
        flash("Access Denied!")
        return redirect(url_for("login"))

    return render_template(
        "officer_dashboard.html",
        username=session["user_name"]
    )


@app.route("/analyst")
def analyst_dashboard():

    if "user_id" not in session:
        return redirect(url_for("login"))

    if session["role"] != "analyst":
        flash("Access Denied!")
        return redirect(url_for("login"))

    return render_template(
        "analyst_dashboard.html",
        username=session["user_name"]
    )
@app.route("/prediction_history")
def prediction_history():

    if "user_id" not in session:
        return redirect(url_for("login"))

    predictions = Prediction.query.filter_by(
        user_id=session["user_id"]
    ).order_by(
        Prediction.prediction_date.desc()
    ).all()

    return render_template(
        "prediction_history.html",
        predictions=predictions
    )
# ============================================================
# Manage Users (Admin Only)
# ============================================================

@app.route("/manage_users")
def manage_users():

    # Check login
    if "user_id" not in session:
        return redirect(url_for("login"))

    # Allow only admin
    if session["role"] != "admin":
        flash("Access Denied!")
        return redirect(url_for("login"))

    # Get all users
    users = User.query.all()

    return render_template(
        "manage_users.html",
        users=users
    )
# ============================================================
# Delete User
# ============================================================

@app.route("/delete_user/<int:user_id>")
def delete_user(user_id):

    if "user_id" not in session:
        return redirect(url_for("login"))

    if session["role"] != "admin":
        flash("Access Denied!")
        return redirect(url_for("login"))

    user = User.query.get_or_404(user_id)

    # Prevent deleting admins
    if user.role == "admin":
        flash("Admin accounts cannot be deleted.")
        return redirect(url_for("manage_users"))

    db.session.delete(user)
    db.session.commit()

    flash("User deleted successfully!")

    return redirect(url_for("manage_users"))
if __name__ == "__main__":
    app.run(debug=True)


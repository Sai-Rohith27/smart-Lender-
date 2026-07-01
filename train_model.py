# ============================================================
# Smart Lender - Loan Eligibility Prediction Model Training
# ============================================================
#

#
# Purpose:
# This script loads the loan dataset, preprocesses the data,
# trains multiple machine learning models, compares their
# performance, and saves the best-performing model.
# ============================================================

# =========================
# Import Required Libraries
# =========================

import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from xgboost import XGBClassifier


# ============================================================
# Load Dataset
# ============================================================

def load_dataset(file_path):
    """
    Load the loan dataset.

    Parameters:
        file_path (str): Path to the CSV dataset.

    Returns:
        pandas.DataFrame: Loaded dataset.
    """
    loan_data = pd.read_csv(file_path)
    return loan_data


# ============================================================
# Preprocess Dataset
# ============================================================

def preprocess_data(loan_data):
    """
    Handle missing values, encode categorical variables,
    and separate features from the target variable.

    Parameters:
        loan_data (DataFrame): Original dataset.

    Returns:
        X (DataFrame): Feature matrix.
        y (Series): Target variable.
    """

    # Fill missing values in categorical columns
    loan_data["Gender"] = loan_data["Gender"].fillna(
        loan_data["Gender"].mode()[0]
    )

    loan_data["Married"] = loan_data["Married"].fillna(
        loan_data["Married"].mode()[0]
    )

    loan_data["Dependents"] = loan_data["Dependents"].fillna(
        loan_data["Dependents"].mode()[0]
    )

    loan_data["Self_Employed"] = loan_data["Self_Employed"].fillna(
        loan_data["Self_Employed"].mode()[0]
    )

    # Fill missing values in numerical columns
    loan_data["LoanAmount"] = loan_data["LoanAmount"].fillna(
        loan_data["LoanAmount"].median()
    )

    loan_data["Loan_Amount_Term"] = loan_data["Loan_Amount_Term"].fillna(
        loan_data["Loan_Amount_Term"].median()
    )

    loan_data["Credit_History"] = loan_data["Credit_History"].fillna(
        loan_data["Credit_History"].median()
    )

    # Encode target variable
    loan_data["Loan_Status"] = loan_data["Loan_Status"].map(
        {
            "N": 0,
            "Y": 1
        }
    )

    # Separate features and target
    X = loan_data.drop(
        columns=["Loan_ID", "Loan_Status"]
    )

    y = loan_data["Loan_Status"]

    # One-Hot Encoding
    X = pd.get_dummies(X)

    return X, y


# ============================================================
# Split Dataset
# ============================================================

def split_dataset(X, y):
    """
    Split dataset into training and testing sets.
    """

    return train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )


# ============================================================
# Scale Features
# ============================================================

def scale_features(X_train, X_test):
    """
    Standardize features for distance-based algorithms
    such as KNN.
    """

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled


# ============================================================
# Train Machine Learning Models
# ============================================================

def train_models(
    X_train,
    X_test,
    y_train,
    y_test,
    X_train_scaled,
    X_test_scaled
):
    """
    Train multiple machine learning models and compare
    their accuracy.
    """

    model_results = {}

    # Decision Tree
    decision_tree_model = DecisionTreeClassifier(
        random_state=42
    )

    decision_tree_model.fit(
        X_train,
        y_train
    )

    decision_tree_accuracy = accuracy_score(
        y_test,
        decision_tree_model.predict(X_test)
    )

    model_results["Decision Tree"] = decision_tree_accuracy

    # KNN
    knn_model = KNeighborsClassifier()

    knn_model.fit(
        X_train_scaled,
        y_train
    )

    knn_accuracy = accuracy_score(
        y_test,
        knn_model.predict(X_test_scaled)
    )

    model_results["KNN"] = knn_accuracy

    # XGBoost
    xgboost_model = XGBClassifier(
        eval_metric="logloss",
        random_state=42
    )

    xgboost_model.fit(
        X_train,
        y_train
    )

    xgboost_accuracy = accuracy_score(
        y_test,
        xgboost_model.predict(X_test)
    )

    model_results["XGBoost"] = xgboost_accuracy

    # Random Forest
    random_forest_model = RandomForestClassifier(
        random_state=42,
        n_estimators=200
    )

    random_forest_model.fit(
        X_train,
        y_train
    )

    random_forest_accuracy = accuracy_score(
        y_test,
        random_forest_model.predict(X_test)
    )

    model_results["Random Forest"] = random_forest_accuracy

    return random_forest_model, model_results


# ============================================================
# Save Trained Model
# ============================================================



def save_model(model, feature_names):
    """
    Save the trained Random Forest model.
    """
    os.makedirs("models", exist_ok=True)

    joblib.dump(
        model,
        "models/loan_model.pkl"
    )

    joblib.dump(
        feature_names,
        "models/feature_names.pkl"
    )

    print("\nModel saved successfully!")
# ============================================================
# Main Function
# ============================================================

def main():

    print("=" * 60)
    print("Smart Lender - Loan Eligibility Prediction")
    print("=" * 60)

    # Load dataset
    loan_data = load_dataset(
        "data/train_u6lujuX_CVtuZ9i.csv"
    )

    # Preprocess dataset
    X, y = preprocess_data(
        loan_data
    )

    # ========================================================
    # Display Feature Information
    # ========================================================

    print("\nFeatures used for model training:\n")
    print(X.columns.tolist())

    print("\nTotal Number of Features:", len(X.columns))

    # Split dataset
    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = split_dataset(
        X,
        y
    )

    # Scale features
    (
        X_train_scaled,
        X_test_scaled
    ) = scale_features(
        X_train,
        X_test
    )

    # Train models
    (
        best_model,
        model_results
    ) = train_models(
        X_train,
        X_test,
        y_train,
        y_test,
        X_train_scaled,
        X_test_scaled
    )

    print("\nModel Comparison")
    print("-" * 30)

    for model_name, accuracy in model_results.items():
        print(f"{model_name:<20}: {accuracy:.4f}")

    # Save the best model
    save_model(
    best_model,
    X.columns.tolist()
)


# ============================================================
# Program Entry Point
# ============================================================

if __name__ == "__main__":
    main()
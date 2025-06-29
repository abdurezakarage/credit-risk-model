import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Load final dataset with is_high_risk target
final_df = pd.read_csv("data/final_with_high_risk.csv")

# Prepare features and target
X = final_df.drop(columns=["is_high_risk", "CustomerId", "TransactionId", "TransactionStartTime"])
y = final_df["is_high_risk"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define models and parameters
models = {
    "logistic_regression": LogisticRegression(max_iter=1000),
    "random_forest": RandomForestClassifier(random_state=42),
    "decision_tree": DecisionTreeClassifier(random_state=42),
    "gradient_boosting": GradientBoostingClassifier(random_state=42)
}

params = {
    "logistic_regression": {
        "C": [0.01, 0.1, 1, 10],
        "solver": ["liblinear", "lbfgs"]
    },
    "random_forest": {
        "n_estimators": [100, 200],
        "max_depth": [5, 10, None]
    },
    "decision_tree": {
        "max_depth": [5, 10, 20, None],
        "min_samples_split": [2, 5, 10]
    },
    "gradient_boosting": {
        "n_estimators": [100, 200],
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 5, 10]
    }
}
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)

# ----------------------------
# Step 1: Load Data
# ----------------------------
def load_data(filepath):
    df = pd.read_csv(filepath)
    X = df.drop(columns=["is_high_risk", "CustomerId", "TransactionId", "TransactionStartTime"])
    y = df["is_high_risk"]
    return X, y

# ----------------------------
# Step 2: Define Model Config
# ----------------------------
def get_model_config():
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        # "random_forest": RandomForestClassifier(random_state=42, class_weight="balanced"),
        # "decision_tree": DecisionTreeClassifier(random_state=42, class_weight="balanced"),
        # "gradient_boosting": GradientBoostingClassifier(random_state=42)
    }

    params = {
        "logistic_regression": {
            "clf__C": [0.01, 0.1, 1, 10],
            "clf__solver": ["liblinear", "lbfgs"]
        },
        "random_forest": {
            "clf__n_estimators": [100, 200],
            "clf__max_depth": [5, 10, None]
        },
        "decision_tree": {
            "clf__max_depth": [5, 10, 20, None],
            "clf__min_samples_split": [2, 5, 10]
        },
        "gradient_boosting": {
            "clf__n_estimators": [100, 200],
            "clf__learning_rate": [0.01, 0.1, 0.2],
            "clf__max_depth": [3, 5, 10]
        }
    }

    return models, params

# ----------------------------
# Step 3: Preprocessor Builder
# ----------------------------
def build_preprocessor(X):
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ])
    return preprocessor

# ----------------------------
# Step 4: Train & Log Best Model (no registry)
# ----------------------------
def train_and_log_best_model(X_train, X_test, y_train, y_test, models, params):
    mlflow.set_experiment("credit-risk-modeling")

    best_f1 = -1.0
    best_model_name = None

    for model_name, model in models.items():
        print(f"\n🔍 Training model: {model_name}")
        with mlflow.start_run(run_name=model_name):
            preprocessor = build_preprocessor(X_train)

            pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("clf", model)
            ])

            grid_search = GridSearchCV(
                pipeline,
                param_grid=params[model_name],
                cv=5,
                scoring="f1",
                n_jobs=-1
            )

            grid_search.fit(X_train, y_train)
            y_pred = grid_search.predict(X_test)
            y_proba = grid_search.predict_proba(X_test)[:, 1]

            # Log metrics and parameters
            mlflow.log_param("model_name", model_name)
            mlflow.log_params(grid_search.best_params_)
            mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
            mlflow.log_metric("precision", precision_score(y_test, y_pred))
            mlflow.log_metric("recall", recall_score(y_test, y_pred))
            mlflow.log_metric("f1_score", f1_score(y_test, y_pred))
            mlflow.log_metric("roc_auc", roc_auc_score(y_test, y_proba))

            print(classification_report(y_test, y_pred))

            # Log the trained model (no registration)
            mlflow.sklearn.log_model(grid_search.best_estimator_, f"{model_name}_model")

            # Track best F1 score
            current_f1 = f1_score(y_test, y_pred)
            if current_f1 > best_f1:
                best_f1 = current_f1
                best_model_name = model_name

    if best_model_name:
        print(f"\n✅ Best model: {best_model_name} (F1 = {best_f1:.4f}) — model logged in MLflow but not registered.")


# ----------------------------
# Step 5: Run the Script
# ----------------------------
if __name__ == "__main__":
    X, y = load_data("data/final_with_high_risk.csv")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    models, params = get_model_config()
    train_and_log_best_model(X_train, X_test, y_train, y_test, models, params)

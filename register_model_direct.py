import mlflow
import os

# Set the MLflow tracking URI to point to the MLflow server
mlflow.set_tracking_uri("http://localhost:5000")

# Path to the model artifacts
model_path = "mlruns/577874245863277567/models/m-21c13abd7e494faca23715f51025316c/artifacts"
model_name = "CreditRiskModel"

try:
    # Register the model directly from the local path
    result = mlflow.register_model(
        model_uri=f"file://{os.path.abspath(model_path)}",
        name=model_name
    )
    print(f"Model registered successfully: {result}")
    
    # Transition to Production
    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(
        name=model_name,
        version=1,
        stage="Production"
    )
    print(f"Model {model_name} version 1 transitioned to Production stage")
    
except Exception as e:
    print(f"Error: {e}") 
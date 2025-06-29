from fastapi import FastAPI
from src.api.pydantic_models import CustomerFeatures, PredictionResponse
import mlflow.pyfunc

app = FastAPI()

# Load the model from MLflow registry
model = mlflow.pyfunc.load_model(model_uri="models:/CreditRiskModel/Production")

@app.post("/predict", response_model=PredictionResponse)
def predict(features: CustomerFeatures):
    input_df = features.to_df()
    prediction = model.predict(input_df)
    return PredictionResponse(risk_probability=prediction[0])

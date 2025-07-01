from fastapi import FastAPI
from src.api.pydantic_models import CustomerFeatures, PredictionResponse
import mlflow.pyfunc
import os

app = FastAPI()

# Load the model directly from the local path
model_path = "mlruns/577874245863277567/models/m-21c13abd7e494faca23715f51025316c/artifacts"
model = mlflow.pyfunc.load_model(model_uri=f"file://{os.path.abspath(model_path)}")

@app.post("/predict", response_model=PredictionResponse)
def predict(features: CustomerFeatures):
    input_df = features.to_df()
    prediction = model.predict(input_df)
    return PredictionResponse(risk_probability=prediction[0])

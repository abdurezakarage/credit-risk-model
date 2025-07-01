from pydantic import BaseModel
import pandas as pd

class CustomerFeatures(BaseModel):
    amount: float
    frequency: int
    recency_days: int
    monetary_value: float

    def to_df(self):
        return pd.DataFrame([self.dict()])

class PredictionResponse(BaseModel):
    risk_probability: float

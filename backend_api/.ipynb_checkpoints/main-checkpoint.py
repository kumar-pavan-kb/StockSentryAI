from fastapi import FastAPI
import pandas as pd
from pydantic import BaseModel

class ClientData(BaseModel):
    AGE_YEARS: float
    ANNUAL_INCOME: float
    ACTIVITY_LEVEL: float

app = FastAPI()

@app.get("/")
def read_root():
    return {"status": "Datathon API is running on LinuxONE!"}

@app.get("/data")
def get_data():
    # Example: Reading the clients.csv file from the parent directory
    try:
        df = pd.read_csv('../clients.csv')
        return {"total_clients": len(df), "columns": list(df.columns)}
    except Exception as e:
        return {"error": str(e)}
@app.post("/predict")
def predict_churn(client: ClientData):
    risk_score = 0.85 if client.ANNUAL_INCOME < 50000 else 0.15
    prediction = "Churn" if risk_score > 0.5 else "Retain"
    
    return {
        "received_data": client.model_dump(),
        "churn_probability": risk_score,
        "prediction": prediction
    }
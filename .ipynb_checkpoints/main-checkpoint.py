from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import os
import pandas as pd

app = FastAPI(title="StockSentry API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "ml_training/stocksentry_model.pkl"

model = None
if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
else:
    print("Model file not found. Running in Mock Mode.")

# Nested Pydantic schemas matching the UI team's payload
class SupplyChainSchema(BaseModel):
    country: str
    shipment_mode: str
    vendor_inco_term: str = "DEX"
    sub_classification: str = "Adult"
    unit_of_measure_per_pack: int = 30
    line_item_quantity: int = 2000
    line_item_value: float
    pack_price: float = 12.5
    unit_price: float = 0.41
    freight_cost_usd: float

class FinancialsSchema(BaseModel):
    current_ratio: float
    debt_to_equity: float
    operating_margin: float = 0.15
    revenue_growth: float = 0.05
    free_cash_flow: float

class RiskPredictionRequest(BaseModel):
    supply_chain: SupplyChainSchema
    financials: FinancialsSchema
@app.get("/health")
def health_check():
    return {
        "status": "online",
        "model_loaded": model is not None
    }

@app.post("/predict/stock-risk")
def predict_stock_risk(data: RiskPredictionRequest):
    if model is None:
        return {
            "risk_level": "MEDIUM", "risk_score": 0.42, "delay_probability": 0.35,
            "recommended_action": "Standard monitoring enabled."
        }
    try:
        # 1. Map lowercase API payload to the EXACT Title Case names the model expects
        flat_data = {
            "Country": data.supply_chain.country,
            "Shipment Mode": data.supply_chain.shipment_mode,
            "Vendor": data.supply_chain.vendor_inco_term,
            "Product Group": data.supply_chain.sub_classification,
            "Line Item Quantity": data.supply_chain.line_item_quantity,
            "Pack Price": data.supply_chain.pack_price,
            "Unit Price": data.supply_chain.unit_price
        }
        input_df = pd.DataFrame([flat_data])
        
        # 2. One-hot encode string columns
        input_df = pd.get_dummies(input_df)
        
        # 3. Align DataFrame with the model's exact expected training features
        if hasattr(model, "feature_names_in_"):
            expected_features = model.feature_names_in_
            
            # Add missing columns filled with 0
            for col in expected_features:
                if col not in input_df.columns:
                    input_df[col] = 0
                    
            # Drop unexpected columns and order them correctly
            input_df = input_df[expected_features]
            
        probability = model.predict_proba(input_df)[0][1] if hasattr(model, "predict_proba") else 0.5
        level = "HIGH" if probability > 0.7 else ("MEDIUM" if probability > 0.4 else "LOW")
        
        return {
            "risk_level": level,
            "risk_score": float(probability),
            "delay_probability": float(probability * 0.8),
            "recommended_action": "Expedite shipment via air freight." if level == "HIGH" else "Proceed with standard logistics."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
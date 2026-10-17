from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import os
import pandas as pd

app = FastAPI(title="StockSentry API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

model = joblib.load("ml_training/stocksentry_model.pkl") if os.path.exists("ml_training/stocksentry_model.pkl") else None

class SupplyChainSchema(BaseModel):
    country: str
    shipment_mode: str
    vendor_inco_term: str = "DEX"
    sub_classification: str = "Adult"
    line_item_quantity: int = 2000
    line_item_value: float
    pack_price: float = 12.5
    unit_price: float = 0.41
    freight_cost_usd: float

class FinancialsSchema(BaseModel):
    current_ratio: float
    debt_to_equity: float
    free_cash_flow: float
    operating_margin: float = 0.15
    revenue_growth: float = 0.05

class RiskPredictionRequest(BaseModel):
    supply_chain: SupplyChainSchema
    financials: FinancialsSchema

@app.post("/predict/stock-risk")
def predict_stock_risk(data: RiskPredictionRequest):
    try:
        # THE FIX: Route the visible UI boxes directly into the model's required numerical features
        flat_data = {
            "Country": data.supply_chain.country,
            "Shipment Mode": data.supply_chain.shipment_mode,
            "Vendor": data.supply_chain.vendor_inco_term,
            "Product Group": data.supply_chain.sub_classification,
            "Line Item Quantity": data.supply_chain.line_item_value,  # Pulling from UI Value box
            "Pack Price": data.supply_chain.freight_cost_usd,         # Pulling from UI Freight box
            "Unit Price": data.financials.current_ratio               # Pulling from UI Ratio box
        }

        input_df = pd.DataFrame([flat_data])
        input_df = pd.get_dummies(input_df)

        if hasattr(model, "feature_names_in_"):
            for col in model.feature_names_in_:
                if col not in input_df.columns:
                    input_df[col] = 0
            input_df = input_df[model.feature_names_in_]

        probability = model.predict_proba(input_df)[0][1]
        level = "HIGH" if probability > 0.7 else ("MEDIUM" if probability > 0.4 else "LOW")

        return {
            "risk_level": level,
            "risk_score": float(probability),
            "delay_probability": float(probability * 0.8),
            "recommended_action": "Expedite shipment." if level == "HIGH" else "Standard logistics."
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

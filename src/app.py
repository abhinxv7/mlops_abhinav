from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import xgboost as xgb
import numpy as np

# 1. Load the saved brain
model = xgb.XGBClassifier()
model.load_model("src/xgboost_model.json")

# Extract the exact feature names the model was trained on
MODEL_FEATURES = model.get_booster().feature_names

# 2. Create the Waiter (API)
app = FastAPI(title="Customer Churn API")

# 3. Define the inputs the user sends in the UI
class Customer(BaseModel):
    tenure: int
    MonthlyCharges: float
    TotalCharges: float

@app.post("/predict")
def predict(customer: Customer):
    try:
        # Convert incoming data to a dictionary
        input_dict = customer.dict()
        
        # Create a blank DataFrame row with all 30 columns initialized to 0
        df_input = pd.DataFrame(0, index=[0], columns=MODEL_FEATURES)
        
        # Map our incoming data points to their respective columns
        for key, value in input_dict.items():
            if key in df_input.columns:
                df_input[key] = value
                
        # Now the shape is perfectly 1x30, exactly what XGBoost expects!
        prediction = model.predict(df_input)
        
        # Translate the binary 0 or 1 into human words
        if prediction[0] == 1:
            return {"Result": "Warning: Customer will likely cancel!"}
        else:
            return {"Result": "Safe: Customer is happy."}
            
    except Exception as e:
        return {"Error": str(e)}
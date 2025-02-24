# from fastapi import FastAPI, HTTPException
# import joblib
# import numpy as np

# app = FastAPI()

# # Charger le modèle
# MODEL_PATH = "model.pkl"
# try:
#     model = joblib.load(MODEL_PATH)
# except Exception as e:
#     raise RuntimeError(f"Erreur lors du chargement du modèle : {e}")

# @app.post("/predict/")
# async def predict(data: dict):
#     try:
#         features = np.array(data["features"]).reshape(1, -1)  # Assurez-vous que les données sont de la bonne forme
#         prediction = model.predict(features)
#         return {"prediction": prediction.tolist()}
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=f"Erreur de prédiction : {e}")

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Define the input data model
class CustomerFeatures(BaseModel):
    account_length: float
    international_plan: int
    voice_mail_plan: int
    number_vmail_messages: float
    total_day_minutes: float
    total_day_calls: float
    total_day_charge: float
    total_eve_minutes: float
    total_eve_calls: float
    total_night_minutes: float
    total_night_calls: float
    total_intl_minutes: float
    total_intl_calls: float
    customer_service_calls: float

# Load the model
MODEL_PATH = "model.pkl"
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

@app.post("/predict/")
async def predict(customer: CustomerFeatures):
    try:
        # Convert input data to numpy array in the correct order
        input_data = np.array([
            customer.account_length,
            customer.international_plan,
            customer.voice_mail_plan,
            customer.number_vmail_messages,
            customer.total_day_minutes,
            customer.total_day_calls,
            customer.total_day_charge,
            customer.total_eve_minutes,
            customer.total_eve_calls,
            customer.total_night_minutes,
            customer.total_night_calls,
            customer.total_intl_minutes,
            customer.total_intl_calls,
            customer.customer_service_calls
        ]).reshape(1, -1)
       
        # Make prediction
        prediction = model.predict(input_data)
       
        return {"prediction": int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Churn Prediction API - Use POST /predict with customer data"}



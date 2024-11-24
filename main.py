from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import requests
import os
from typing import List, Dict
import logging
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from contextlib import asynccontextmanager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variable for model
model = None

@asynccontextmanager
async def lifespan_handler(app: FastAPI):
    """Load model on startup instead of module level"""
    global model
    try:
        model_url = 'https://github.com/Austin-Z/ChurnML/raw/refs/heads/main/xgb_model.joblib'
        logger.info("Starting model download...")

        response = requests.get(model_url, timeout=30, allow_redirects=True)  # Add timeout
        response.raise_for_status()  # Raise error for bad status codes
        
        logger.info("Model download completed. Saving file...")

        model_file = 'xgb_model.joblib'

        with open(model_file, 'wb') as f:
            f.write(response.content)
        
        model = joblib.load(model_file)
        logger.info("Model loaded successfully")

    except Exception as e:
        if 'requests' in str(e):
            logger.error("Error during model download")
            raise HTTPException(status_code=500, detail="Failed to download model")
        else:
            logger.error(f"Unhandled error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
    
    yield

app = FastAPI(title="ChurnML Predictor",lifespan=lifespan_handler)
# Enable CORS for all origins 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy"}

@app.post("/predict-churn")
async def predict_churn(file: UploadFile = File(...)) -> Dict:
    """
    Predict churn probability from uploaded CSV file
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed")
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Read the uploaded CSV file into a pandas DataFrame
        df = pd.read_csv(file.file)

        df = df.dropna(subset=['Total Charges'])
        # Store customerID separately
        customer_ids = df['CustomerID']

        df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce')
        df = df.drop(['Count', 'Country', 'State', 'City', 'Zip Code', 'Lat Long', 'Latitude', 'Longitude', 'Churn Label', 'Churn Score', 'Churn Reason'], axis=1)
        df['ChargesPerMonth'] = df['Total Charges'] / df['Tenure Months']
        df = df.drop(['Total Charges', 'Monthly Charges'], axis=1)
        df = df.drop(['Churn Value','CustomerID'], axis=1)
        df = pd.get_dummies(df)

        model_features = model.feature_names_in_
        missing_features = [col for col in model_features if col not in df.columns]
        if missing_features:
            raise HTTPException(
                status_code=400,
                detail=f"Input data is missing required feature columns: {', '.join(missing_features)}"
            )
        # Separate features for model prediction
        df_features = df.reindex(columns=model_features, fill_value=0)
        
        # Predict churn probabilities

        churn_probabilities = model.predict_proba(df_features)[:, 1]
        
        df['CustomerID'] = customer_ids
        # Prepare response
        results = sorted(
            [
                {
                    "CustomerID": str(customer_ids),  # Original CustomerID for display
                    "CLTV": float(cltv),
                    "ChargesPerMonth": float(charges_per_month),
                    "Churn Probabilities": float(prob)
                }
                for customer_ids, cltv, charges_per_month, prob in zip(
                    df['CustomerID'], df['CLTV'], df['ChargesPerMonth'], churn_probabilities
                )
            ],
            key=lambda x: x['Churn Probabilities'],
            reverse=True
        )
        
        return {"predictions": results}
        
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="The uploaded CSV file is empty")
    except pd.errors.ParserError:
        raise HTTPException(status_code=400, detail="Invalid CSV format")
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

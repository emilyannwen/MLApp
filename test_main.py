import requests
import os
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

# Test the /health endpoint
def test_health_check():
    response = requests.get("http://127.0.0.1:8000/health")
    assert response.status_code == 200, "Health check endpoint failed."
    assert response.json() == {"status": "healthy"}

# Test the /predict-churn endpoint
def test_predict_churn():

    test_file_path = r"C:\Users\austi\Github\LocalHost\Telco_customer_churn_Testing.csv"

    assert os.path.exists(test_file_path), f"Sample data file not found at {test_file_path}."

    with open(test_file_path, "rb") as file:
        response = requests.post(
            "http://127.0.0.1:8000/predict-churn",
            files={"file": ("Telco_customer_churn_Testing.csv", file, "text/csv")}
        )

    assert response.status_code == 200, f"Prediction endpoint failed with status code {response.status_code}."
    data = response.json()

    assert "predictions" in data
    for prediction in data["predictions"]:
        assert "CustomerID" in prediction
        assert "CLTV" in prediction
        assert "ChargesPerMonth" in prediction
        assert "Churn Probabilities" in prediction 
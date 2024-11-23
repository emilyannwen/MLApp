import io
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock
from main import app
import pandas as pd
from unittest.mock import patch
import os


@pytest.fixture
def raw_csv_file():
    """Read a static CSV file from the resources folder."""
    csv_path = os.path.join(os.path.dirname(__file__), "resources/Telco_customer_churn_Testing.csv")
    with open(csv_path, "rb") as f:  # Open the file in binary mode
        return f.read()

@pytest.fixture
def test_client():
    """Ensure the app's startup event runs before the test."""
    client = TestClient(app)

    # Use `with` block to ensure the app lifecycle is managed
    with client:
        # Trigger the startup event manually to ensure the model is loaded
        client.app.router.startup()
        yield client  # Provide the client for use in the test

@pytest.fixture
def mock_model():
    """Mock the model loading process to avoid downloading the real model."""
    with patch("main.model", "mocked_model"):  # Replace the global `model` with a mock
        yield

def test_predict_churn_with_startup(test_client, raw_csv_file, mock_model):
    """Test the happy path for the /predict-churn endpoint."""
    response = test_client.post(
        "/predict-churn",
        files={"file": ("test.csv", raw_csv_file, "text/csv")},
    )

    # Assert the response status code
    assert response.status_code == 200

    # Parse the response JSON
    json_resp = response.json()

    # Validate the response structure and predictions
    assert "predictions" in json_resp
    predictions = json_resp["predictions"]
    assert len(predictions) == 2  # Ensure there are 2 predictions (for 2 rows in the CSV)

    # Check the content of the first prediction
    assert predictions[0]["CustomerID"] == "3668-QPYBK"
    assert predictions[0]["CLTV"] == 3239
    assert predictions[0]["Churn Probabilities"] == 0.7  # Mocked churn probability

    # Check the content of the second prediction
    assert predictions[1]["CustomerID"] == "9237-HQITU"
    assert predictions[1]["CLTV"] == 2701
    assert predictions[1]["Churn Probabilities"] == 0.7  # Mocked churn probability
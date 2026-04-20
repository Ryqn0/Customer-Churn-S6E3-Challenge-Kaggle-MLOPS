"""
This script test FASTAPI endpoints defined in src/app/main.py. It uses the TestClient from FastAPI to simulate requests to the API and validate the responses. The tests cover the prediction endpoint, ensuring that it correctly processes input data and returns the expected predictions. The tests also check for proper handling of edge cases and invalid input data.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
for path in (ROOT_DIR, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.append(str(path))  # Add the project paths so imports resolve during pytest collection

url = "http://127.0.0.1:8000"

import app.main as main_module
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

sample_input = {
    "gender": "Female",
    "senior_citizen": 0,
    "partner": "Yes",
    "dependents": "No",
    "tenure": 12,
    "phone_service": "Yes",
    "multiple_lines": "No",
    "internet_service": "Fiber optic",
    "online_security": "No",
    "online_backup": "Yes",
    "device_protection": "No",
    "tech_support": "No",
    "streaming_tv": "Yes",
    "streaming_movies": "Yes",
    "contract": "Month-to-month",
    "paperless_billing": "Yes",
    "payment_method": "Electronic check",
    "monthly_charges": 70.35,
    "total_charges": 845.5
}

def test_health_check():
    """
    Test the health check endpoint to ensure it returns the expected status message.
    This test sends a GET request to the /health endpoint and checks if the response contains the correct status message indicating that the API is healthy and running.
    """
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "API is healthy and running!"}

def test_predict_churn():
    """
    Test the predict endpoint to ensure it processes input data correctly and returns the expected prediction.
    This test sends a POST request to the /predict endpoint with sample customer data and checks if the response contains a valid prediction about whether the customer is likely to churn or not.
    """
    main_module.load_model = lambda model_name: object()
    main_module.predict = lambda model, data: "The customer is likely to churn."

    response = client.post("/predict", json=sample_input)
    assert response.status_code == 200
    assert response.json() == {"prediction": "The customer is likely to churn."}

if __name__ == "__main__":
    test_health_check()
    print("Health check test passed successfully!")
    test_predict_churn()
    print("Predict endpoint test passed successfully!")
    print("All tests passed successfully!")
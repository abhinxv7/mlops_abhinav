from fastapi.testclient import TestClient
from src.app import app

# Create a fake user to test our API
client = TestClient(app)

def test_predict():
    # Send fake customer data to the Waiter
    response = client.post("/predict", json={"tenure": 5, "MonthlyCharges": 50.0, "TotalCharges": 250.0})
    
    # Check if the Waiter responded successfully (Status 200 means OK)
    assert response.status_code == 200
    # Check if the Waiter gave us a "Result"
    assert "Result" in response.json()
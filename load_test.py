
from locust import HttpUser, task, between

class ChurnPredictionUser(HttpUser):
    # Define wait time between requests to simulate realistic user behavior
    wait_time = between(1, 3)  # Wait between 1 to 3 seconds randomly

    @task
    def predict_churn(self):
        # Simulate a churn prediction request with a sample CSV
        with open("Telco_customer_churn_Testing.csv", "rb") as file:
            self.client.post("/predict-churn", files={"file": ("Telco_customer_churn_Testing.csv", file, "text/csv")})

# Run the test with 100 users and a spawn rate of 10 users per second
# Use the command line: locust -f load_test.py --host=http://127.0.0.1:8000 --users 100 --spawn-rate 10 --run-time 1m


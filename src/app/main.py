"""
This file contains the main entry point for the application.
It uses FastAPI to create a web server that exposes an endpoint for making predictions using a pre-trained model.
It uses GRADIO to create a user interface for the prediction endpoint, allowing users to input data and receive predictions in a user-friendly way.
"""

from fastapi import FastAPI
from pydantic import BaseModel
import gradio as gr
from models.run_inference_pipeline import load_model, predict

# Initialize the FastAPI app
app = FastAPI(
    title="Customer Churn Prediction API",
    description="An API for predicting customer churn using a pre-trained model.",
    version="1.0.0"
)

# HEALTH CHECK ENDPOINT
@app.get("/health")
def health_check():
    """
    Health check endpoint to verify that the API is running.
    Returns a simple message indicating the status of the API.
    """
    return {"status": "API is healthy and running!"}

# DATA SCHEMA FOR PREDICTION
class CustomerData(BaseModel):
    """
    Pydantic model for validating the input data for predictions.
    This model defines the expected structure of the input data, including the required fields and their types.
    """
    gender: str # "Male" or "Female"
    senior_citizen: int # 0 or 1
    partner: str # "Yes" or "No"
    dependents: str # "Yes" or "No"
    tenure: int # Number of months the customer has been with the company
    phone_service: str # "Yes" or "No"
    multiple_lines: str # "Yes", "No", or "No phone service"
    internet_service: str # "DSL", "Fiber optic", or "No"
    online_security: str # "Yes", "No", or "No internet service"
    online_backup: str # "Yes", "No", or "No internet service"
    device_protection: str # "Yes", "No", or "No internet service"
    tech_support: str # "Yes", "No", or "No internet service"
    streaming_tv: str # "Yes", "No", or "No internet service"
    streaming_movies: str # "Yes", "No", or "No internet service"
    contract: str # "Month-to-month", "One year", or "Two year"
    paperless_billing: str # "Yes" or "No"
    payment_method: str # "Electronic check", "Mailed check", "Bank transfer (automatic)", or "Credit card (automatic)"
    monthly_charges: float # Monthly charges for the customer
    total_charges: float # Total charges for the customer

# PREDICTION ENDPOINT
@app.post("/predict")
def predict_churn(customer_data: CustomerData):
    """
    Endpoint for making predictions about customer churn.
    This endpoint accepts a POST request with customer data in the request body, validates the input data, and returns a prediction about whether the customer is likely to churn or not.
    """
    # Load the pre-trained model
    model = load_model("voting_model")

    # Convert the input data to a dictionary
    input_data = customer_data.dict()

    # Make a prediction using the loaded model
    prediction = predict(model, input_data)

    return {"prediction": prediction}

# GRADIO INTERFACE
def gradio_interface(gender: str, senior_citizen: bool, partner: str, dependents: str, tenure: int, phone_service: str, 
                    multiple_lines: str, internet_service: str, online_security: str, online_backup: str, device_protection: str, 
                    tech_support: str, streaming_tv: str, streaming_movies: str, contract: str, paperless_billing: str, payment_method: str, 
                    monthly_charges: float, total_charges: float) -> str:
    """
    Function to create a GRADIO interface for the prediction endpoint.
    This interface allows users to input customer data through a web-based form and receive predictions in a user-friendly way.
    """
    data = {
        "gender": gender,
        "senior_citizen": 1 if senior_citizen else 0,
        "partner": partner,
        "dependents": dependents,
        "tenure": tenure,
        "phone_service": phone_service,
        "multiple_lines": multiple_lines,
        "internet_service": internet_service,
        "online_security": online_security,
        "online_backup": online_backup,
        "device_protection": device_protection,
        "tech_support": tech_support,
        "streaming_tv": streaming_tv,
        "streaming_movies": streaming_movies,
        "contract": contract,
        "paperless_billing": paperless_billing,
        "payment_method": payment_method,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges
    }
    return predict(load_model("voting_model"), data)

# GRADIO UI COMPONENTS
gradio_interface = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.Dropdown(choices=["Male", "Female"], label="Gender"),
        gr.Checkbox(label="Senior Citizen"),
        gr.Dropdown(choices=["Yes", "No"], label="Partner"),
        gr.Dropdown(choices=["Yes", "No"], label="Dependents"),
        gr.Slider(0, 72, step=1, label="Tenure"),
        gr.Dropdown(choices=["Yes", "No"], label="Phone Service"),
        gr.Dropdown(choices=["Yes", "No", "No phone service"], label="Multiple Lines"),
        gr.Dropdown(choices=["DSL", "Fiber optic", "No"], label="Internet Service"),
        gr.Dropdown(choices=["Yes", "No", "No internet service"], label="Online Security"),
        gr.Dropdown(choices=["Yes", "No", "No internet service"], label="Online Backup"),
        gr.Dropdown(choices=["Yes", "No", "No internet service"], label="Device Protection"),
        gr.Dropdown(choices=["Yes", "No", "No internet service"], label="Tech Support"),
        gr.Dropdown(choices=["Yes", "No", "No internet service"], label="Streaming TV"),
        gr.Dropdown(choices=["Yes", "No", "No internet service"], label="Streaming Movies"),
        gr.Dropdown(choices=["Month-to-month", "One year", "Two year"], label="Contract"),
        gr.Dropdown(choices=["Yes", "No"], label="Paperless Billing"),
        gr.Dropdown(choices=["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], label="Payment Method"),
        gr.Number(label="Monthly Charges"),
        gr.Number(label="Total Charges")
    ],
    outputs=gr.Textbox(label="Prediction"),
    title="Customer Churn Prediction",
    description="Enter customer data to predict whether they are likely to churn or not.",
    examples=[
        ["Female", False, "Yes", "No", 12, "Yes", "No", "Fiber optic", "No", "No", "No", "No", "Yes", "No", "Month-to-month", "Yes", "Electronic check", 70.35, 845.5],
        ["Male", True, "No", "Yes", 24, "No", "No phone service", "DSL", "Yes", "Yes", "Yes", "Yes", "No", "No", "Two year", "No", "Bank transfer (automatic)", 99.65, 2399.0]
    ],theme=gr.themes.Soft()
)

# MOUNT THE GRADIO INTERFACE TO THE FASTAPI APP
app = gr.mount_gradio_app(app, gradio_interface, path="/gradio")
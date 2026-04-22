# Customer Churn Prediction — MLOps Project

End-to-end MLOps project built on the [Kaggle Playground Series S6E3](https://www.kaggle.com/competitions/playground-series-s6e3) dataset. The goal is to predict customer churn using an ensemble model, while applying production-grade MLOps practices: experiment tracking, data versioning, containerization, and cloud deployment.

---

## Architecture Overview

```
Training Pipeline          Serving Pipeline              Cloud Deployment
─────────────────          ────────────────              ────────────────
Data Loading               FastAPI REST API        ──►   GCP Cloud Run
Data Validation    ──►     Gradio UI               ──►   Artifact Registry
Preprocessing              Model Inference                (europe-west1)
Feature Engineering
Model Training (VotingClassifier)
Experiment Tracking (MLflow)
Model Serialization (joblib)
```

---

## Tech Stack

| Category | Tools |
|----------|-------|
| **ML / Training** | scikit-learn, XGBoost, LightGBM, CatBoost, Optuna |
| **Experiment Tracking** | MLflow |
| **Data Versioning** | DVC |
| **Data Validation** | Great Expectations |
| **Serving** | FastAPI, Uvicorn, Gradio |
| **Containerization** | Docker |
| **CI** | GitHub Actions |
| **Cloud** | Google Cloud Platform — Cloud Run, Artifact Registry |
| **Testing** | pytest |

---

## Project Structure

```
├── scripts/
│   └── run_training_pipeline.py   # Full training pipeline (load → validate → preprocess → train → evaluate)
├── src/
│   ├── app/
│   │   └── main.py                # FastAPI app + Gradio interface
│   ├── data/
│   │   ├── load_data.py
│   │   └── preprocess.py
│   ├── features/
│   │   └── build_features.py
│   ├── models/
│   │   ├── train.py
│   │   └── evaluate.py
│   └── utils/
│       └── validate_data.py
├── models/
│   ├── run_inference_pipeline.py  # Inference logic (load model, predict)
│   └── voting_model.pkl           # Serialized VotingClassifier
├── data/
│   └── processed/
│       └── feature_names.json     # Training feature schema (used for inference alignment)
├── tests/                         # pytest test suite
├── artifacts/                     # Evaluation reports
├── dockerfile
├── .dockerignore
├── requirements.txt
└── .github/
    └── workflows/
        └── ci.yml                 # GitHub Actions — build & push Docker image
```

---

## Model

The model is a **VotingClassifier** (soft voting) combining:
- **XGBoost** (`XGBClassifier`)
- **LightGBM** (`LGBMClassifier`)
- **CatBoost** (`CatBoostClassifier`)

Hyperparameters tuned with **Optuna**. All experiments tracked with **MLflow** (local SQLite backend).

**Training pipeline steps:**
1. Load data
2. Validate schema with Great Expectations
3. Preprocess (encoding, type casting, renaming)
4. Feature engineering
5. Train VotingClassifier
6. Evaluate (accuracy, precision, recall, F1, ROC-AUC)
7. Log metrics and artifacts to MLflow
8. Serialize model with joblib

---

## API Endpoints

The FastAPI app exposes the following endpoints:

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/check_model` | Verify the model loads correctly |
| `POST` | `/predict` | Predict churn from customer data (JSON) |
| `GET` | `/gradio` | Interactive Gradio web UI |
| `GET` | `/docs` | Auto-generated Swagger documentation |

### Example `/predict` request

```bash
curl -X POST "https://<your-cloud-run-url>/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female",
    "senior_citizen": 0,
    "partner": "Yes",
    "dependents": "No",
    "tenure": 12,
    "phone_service": "Yes",
    "multiple_lines": "No",
    "internet_service": "Fiber optic",
    "online_security": "No",
    "online_backup": "No",
    "device_protection": "No",
    "tech_support": "No",
    "streaming_tv": "Yes",
    "streaming_movies": "No",
    "contract": "Month-to-month",
    "paperless_billing": "Yes",
    "payment_method": "Electronic check",
    "monthly_charges": 70.35,
    "total_charges": 845.5
  }'
```

---

## Local Development

### Prerequisites

- Python 3.13
- Docker Desktop
- `gcloud` CLI (for GCP deployment)

### Run locally

```bash
# Install dependencies
pip install -r requirements.txt

# Train the model
python scripts/run_training_pipeline.py

# Start the API
python -m uvicorn src.app.main:app --host 0.0.0.0 --port 8000
```

### Run with Docker

```bash
docker build -t customer-churn-api -f dockerfile .
docker run -p 8000:8000 customer-churn-api
```

Then open http://localhost:8000/gradio for the UI or http://localhost:8000/docs for the API.

### Run tests

```bash
pytest tests/
```

---

## Deployment — Google Cloud Platform

The application is deployed as a containerized service on **GCP Cloud Run** (region: `europe-west1`).

### Infrastructure

| Resource | Value |
|----------|-------|
| GCP Project | `customer-churn-mlops-s6e3-rd` |
| Region | `europe-west1` (Belgium) |
| Container Registry | Artifact Registry (`customer-churn` repository) |
| Runtime | Cloud Run (serverless, scale-to-zero) |
| Memory | 512 Mi |

### Manual deployment

```bash
# 1. Authenticate Docker with Artifact Registry
gcloud auth print-access-token | docker login -u oauth2accesstoken \
  --password-stdin https://europe-west1-docker.pkg.dev

# 2. Build and push the image
docker build -t europe-west1-docker.pkg.dev/customer-churn-mlops-s6e3-rd/customer-churn/api:v1 -f dockerfile .
docker push europe-west1-docker.pkg.dev/customer-churn-mlops-s6e3-rd/customer-churn/api:v1

# 3. Deploy to Cloud Run
gcloud run deploy customer-churn-api \
  --image europe-west1-docker.pkg.dev/customer-churn-mlops-s6e3-rd/customer-churn/api:v1 \
  --platform managed \
  --region europe-west1 \
  --port 8000 \
  --allow-unauthenticated \
  --memory 512Mi
```

### CI/CD

On every push to `main`, GitHub Actions (`.github/workflows/ci.yml`) automatically builds and pushes the Docker image to Docker Hub.


---

## Data Versioning

Large data files are tracked with **DVC** (not committed to Git). To pull the data:

```bash
dvc pull
```

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `PYTHONPATH` | Set to `/app/src:/app` in the Docker image for correct module resolution |

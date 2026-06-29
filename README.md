# Self-Healing MLOps Pipeline: End-to-End Production Churn Predictor

An enterprise-grade, closed-loop MLOps pipeline designed to train, serve, monitor, and automatically retrain a machine learning model in production. This project addresses a fundamental reality of real-world AI: **models decay over time.** Instead of just serving predictions passively, this system integrates **automated data drift detection** that programmatically triggers a self-healing retraining pipeline the moment production features deviate from baseline statistics.

---

## 🏗️ System Architecture

```
                       +------------------------+
                       |   Incoming Data Stream |
                       +-----------+------------+
                                   |
                                   v
+------------------+     +---------+----------+     +--------------------+
|   FastAPI App    |<----+  Data Drift Monitor|<----+ Baseline reference |
| (The Waiter API) |     |  (Evidently AI)    |     |    (churn.csv)     |
+------------------+     +---------+----------+     +--------------------+
                                   |
                         [Drift Detected? Yes]
                                   |
                                   v
                         +---------+----------+
                         | Automated Retraining|
                         |     (train.py)     |
                         +---------+----------+
                                   |
                                   v
                         +---------+----------+
                         |  MLflow Tracking   |
                         |   & Model Update   |
                         +--------------------+

```

---

## 🚀 Key Features

* **Production Serving:** High-performance REST API built with **FastAPI** and containerized using **Docker** for seamless cloud deployment.
* **Experiment Tracking:** Deep visibility into hyperparameters and model artifacts using **MLflow**.
* **Automated Watchdog (Data Drift):** Real-time monitoring via **Evidently AI** leveraging statistical checks to catch feature shifts.
* **Self-Healing Loop:** Programmatic orchestration using Python's `subprocess` engine to immediately kick off model retraining when data corruption or environmental shifts cross the target threshold (`drift_share=0.01`).
* **Robust CI/CD:** **GitHub Actions** workflow that handles automated unit testing via `pytest` and `TestClient` on every code push.

---

## 📁 Repository Structure

```text
mlops_project/
├── .github/
│   └── workflows/
│       └── ci.yml             # GitHub Actions continuous integration pipeline
├── data/
│   └── churn.csv              # Telco Customer Churn dataset (Reference data)
├── notebooks/                 # For exploratory data analysis (EDA)
├── src/
│   ├── app.py                 # FastAPI application serving predictions
│   ├── monitor.py             # Watchdog script checking drift & triggering retraining
│   ├── train.py               # XGBoost training script integrated with MLflow
│   └── xgboost_model.json     # Serialized active model artifact
├── tests/
│   └── test_main.py           # Automated API endpoint unit tests
├── Dockerfile                 # Containerization instructions for the API
├── docker-compose.yml         # Local environment multi-container orchestration
└── requirements.txt           # Project Python dependencies

```

---

## 🛠️ Tech Stack

* **Core ML:** Python, Pandas, Scikit-Learn, XGBoost
* **MLOps & Monitoring:** MLflow, Evidently AI
* **API & Deployment:** FastAPI, Uvicorn, Docker, Docker Compose
* **Testing & CI/CD:** Pytest, HTTPX, GitHub Actions

---

## ⚡ Getting Started

### Prerequisites

* Python 3.9+
* Docker & Docker Compose (Optional, for containerized run)

### Local Setup & Installation

1. **Clone the repository and navigate inside:**
```bash
git clone <your-repo-url>
cd mlops_project

```


2. **Install dependencies:**
```bash
pip install -r requirements.txt

```


3. **Train the baseline model:**
```bash
python src/train.py

```


4. **Boot up the MLflow Dashboard to inspect your training metrics:**
```bash
mlflow ui

```


*Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.*

---

## 🏃 Running the Application

### 1. Launching the Serving API

Start the FastAPI server locally:

```bash
uvicorn src.app:app --reload

```

* **Interactive API Docs:** Head to [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) to test predictions using the built-in Swagger UI.

### 2. Testing the Self-Healing Loop (Drift Trigger)

To simulate production data decay, `monitor.py` intentionally injects an economic anomaly into 20% of the incoming customer billing profiles.

Execute the watchdog script:

```bash
python src/monitor.py

```

**What happens behind the scenes:**

1. Evidently AI builds a localized metrics analysis suite.
2. It generates a comprehensive diagnostic page (`drift_dashboard.html`).
3. The automated threshold validation identifies feature drift on `MonthlyCharges`.
4. The script sends an internal interrupt, logs a `🚨 ALERT`, and automatically runs `src/train.py` to overwrite the stale model with a freshly adjusted iteration.

### 3. Containerized Orchestration

Spin up the entire production environment inside an isolated container loop:

```bash
docker compose up -d

```

---

## 🧪 Automated Testing

Run the test suite locally to verify API validation schemas and response payloads before shipping:

```bash
pytest tests/

```

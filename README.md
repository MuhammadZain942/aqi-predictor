# Rawalpindi Air Quality Predictor (End-to-End MLOps)

## Internship Project Submission

**Submitted to:** 10Pearls
**Submitted by:** Muhammad Zain
**Role:** DataScience Intern
**Project Type:** End-to-End Serverless Machine Learning Pipeline

---

## Executive Summary

The **Rawalpindi Air Quality Predictor** is a production-grade, end-to-end MLOps system designed to forecast the Air Quality Index (AQI) for Rawalpindi, Pakistan. The project demonstrates a fully automated, serverless machine learning pipeline that continuously ingests real-time data, retrains multiple models, and deploys the best-performing model without manual intervention.

The system is built using modern MLOps best practices, including CI/CD automation, feature stores, model registries, and reproducible training pipelines. Through GitHub Actions and Hopsworks, the solution ensures scalability, reliability, and continuous performance improvement.

---

## Key Performance Metrics

During validation and testing, multiple machine learning algorithms were evaluated. The **Random Forest Regressor** consistently emerged as the top-performing model:

* **Accuracy (R² Score):** Consistently achieved **78% – 80%** on test data
* **Error Rate:** Low Mean Absolute Error (MAE), ensuring reliable and stable predictions
* **Automated Model Selection:** The training pipeline compares Random Forest, Gradient Boosting, and Ridge Regression models daily. Only the highest-performing model is deployed to production

---

## Project Objectives

* **Predictive Capability:** Forecast AQI trends for the next **72 hours (3 days)** to support environmental awareness and decision-making
* **Full Automation:** Implement a serverless MLOps architecture with zero manual intervention for data ingestion, training, and deployment
* **Model Optimization:** Apply a **Champion vs. Challenger** strategy for continuous model improvement
* **Scalability:** Use a cloud-native **Feature Store (Hopsworks)** to efficiently manage and scale historical and real-time data

---

## System Architecture

The project is organized into **three decoupled and automated pipelines**, ensuring modularity and maintainability.

### 1. Feature Pipeline (ETL & Data Ingestion)

* **Execution Frequency:** Hourly (automated using GitHub Actions)
* **Data Source:** Open-Meteo Real-Time Weather API
* **Workflow:**

  * Fetches live environmental variables such as **PM2.5, PM10, Nitrogen Dioxide (NO₂), and Ozone (O₃)**
  * Performs data cleaning, validation, and feature engineering (including time-based features)
  * Ingests processed features into the **Hopsworks Feature Store**, creating a single source of truth

### 2. Training Pipeline (Continuous Training – CT)

* **Execution Frequency:** Daily (automated using GitHub Actions)
* **Workflow:**

  * Retrieves historical training data directly from the Feature Store
  * Trains multiple models in parallel (**Multi-Model Battle**):

    * Random Forest Regressor *(Current Champion: ~80% R²)*
    * Gradient Boosting Regressor
    * Ridge Regression
  * Evaluates models using the **R² Score**
  * Automatically registers the best-performing model in the **Hopsworks Model Registry**
  * Applies **model versioning** for reproducibility and traceability

### 3. Inference Pipeline (Deployment)

* **User Interface:** Streamlit Web Application
* **Workflow:**

  * Loads the latest **Champion Model** from the Model Registry
  * Fetches weather forecast data for the upcoming three days
  * Generates daily average AQI predictions
  * Displays results through an interactive dashboard with AQI health category indicators

---

## Technology Stack

| Component             | Technology     | Purpose                                   |
| --------------------- | -------------- | ----------------------------------------- |
| Programming Language  | Python 3.9     | Core application and ML logic             |
| MLOps Platform        | Hopsworks.ai   | Feature Store and Model Registry          |
| CI/CD & Orchestration | GitHub Actions | Automated pipelines and scheduling        |
| Machine Learning      | Scikit-learn   | Model training and evaluation             |
| Frontend              | Streamlit      | Interactive web dashboard                 |
| Data Processing       | Pandas, Joblib | Data manipulation and model serialization |

---

## Installation & Local Execution

Follow the steps below to run the project locally.

### 1. Clone the Repository

```bash
git clone https://github.com/MuhammadZain942/aqi-predictor.git
cd aqi-predictor
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file in the root directory and add your Hopsworks API credentials:

```env
HOPSWORKS_API_KEY=pk_your_long_api_key_here
```

### 4. Run the Streamlit Dashboard

```bash
python -m streamlit run 3_app.py
```

---

## Project Highlights

* Fully automated **serverless MLOps pipeline**
* Real-time data ingestion with continuous retraining
* Champion–Challenger model selection strategy
* Production-ready deployment with model versioning and monitoring

---

## Author

**Muhammad Zain**
DataScience Intern

Developed as part of the **10Pearls Internship Program**.

# Fraudulent Transaction Detection System

This project predicts whether a financial transaction is fraudulent using a Logistic Regression model.  
It demonstrates a complete data science workflow including experiment tracking, hyperparameter tuning, a Flask-based user interface, and CI setup with GitHub Actions.

---

## Project Overview

The system:
- Trains a Logistic Regression model for fraud detection
- Tracks experiments, parameters, metrics, and artifacts using **MLflow**
- Tunes hyperparameters using **Optuna**
- Serves predictions via a **Flask web application**
- Uses **GitHub Actions** for Continuous Integration

---

## Dataset Features

Input features used by the model:

- `amount`
- `transaction_hour`
- `merchant_category`  
  (Clothing, Electronics, Food, Grocery, Travel)
- `foreign_transaction`
- `location_mismatch`
- `device_trust_score`
- `velocity_last_24h`
- `cardholder_age`

Target:
- `is_fraud`  
  (0 = Legitimate, 1 = Fraudulent)

---

## Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- MLflow
- Optuna
- Flask
- HTML (inline CSS)
- GitHub Actions

---


---

## Model Training and Hyperparameter Tuning

Model training and tuning are handled using **MLflow + Optuna**.

Each Optuna trial:
- Trains a Logistic Regression model
- Logs parameters and metrics to MLflow
- Saves the trained model as an artifact
- Runs as a nested MLflow run under a parent study run

Metrics tracked:
- Precision
- Recall
- F1-score
- Accuracy

Run training:

```bash
python src/train.py

## Installation steps 

git clone https://github.com/your-username/fraud-detection.git
cd fraud-detection

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

uv add -r requirements.txt



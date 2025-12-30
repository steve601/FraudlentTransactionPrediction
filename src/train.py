import pandas as pd
import numpy as np
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score,accuracy_score
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv('data/credit_card_fraud_10k.csv')

data.drop(columns=['transaction_id'], inplace=True)

# Encode categorical variables
encoder = LabelEncoder()
data['merchant_category'] = encoder.fit_transform(data['merchant_category'])

X = data.drop(columns=['is_fraud'])
y = data['is_fraud']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

numeric_cols = ['amount', 'transaction_hour', 'device_trust_score', 'cardholder_age']
scaler = MinMaxScaler()
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

linear_model = LogisticRegression(
    penalty = 'l1',
    C = 0.59,
    solver='liblinear',
    class_weight= None,
    max_iter=200)
linear_model.fit(X_train, y_train)

precision = precision_score(y_test, linear_model.predict(X_test))
recall = recall_score(y_test, linear_model.predict(X_test)) 
f1 = f1_score(y_test, linear_model.predict(X_test))

print("Precision: ", precision)
print("Recall: ", recall)
print("F1 Score: ", f1)

mlflow.sklearn.log_model(linear_model, artifact_path="model")

from flask import Flask, render_template, request
import pandas as pd
import pickle
import os
import mlflow.sklearn

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = mlflow.sklearn.load_model("final_model")
scaler = pickle.load(open(os.path.join(BASE_DIR, "notebook", "scaler.pkl"), "rb"))

@app.route('/')
def homepage():
    return render_template('index.html')            

@app.route('/predict',methods=['POST'])
def predict():
    x = [i for i in request.form.values()]
    columns = ['amount', 'transaction_hour', 'merchant_category',
       'foreign_transaction', 'location_mismatch', 'device_trust_score',
       'velocity_last_24h', 'cardholder_age']
    x = pd.DataFrame([x],columns=columns)

    numeric_cols = ['amount', 'transaction_hour', 'device_trust_score', 'cardholder_age']
    x[numeric_cols] = scaler.transform(x[numeric_cols])

    y_pred = model.predict(x)
    msg = "Fraudulent Transaction" if y_pred[0]==1 else "Legitimate Transaction"
    
    return render_template('index.html',text=msg)

if __name__ == "__main__":
    app.run(host="0.0.0.0",port=5000)
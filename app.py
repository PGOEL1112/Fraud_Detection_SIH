from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import os

app = Flask(__name__)

# ---------- Load existing models if available ----------
rf_model_path = "fraud_model_rf.pkl"
xgb_model_path = "fraud_model_xgb.pkl"

rf_model = joblib.load(rf_model_path) if os.path.exists(rf_model_path) else None
xgb_model = joblib.load(xgb_model_path) if os.path.exists(xgb_model_path) else None


# ---------- Helper function ----------
def train_and_save_models(df):
    try:
        data = df.copy()

        if 'herb_type' in data.columns:
            le = LabelEncoder()
            data['herb_type'] = le.fit_transform(data['herb_type'])

        if 'Fraud' not in data.columns:
            raise ValueError("'Fraud' column not found in dataset. Please add it before training.")

        X = data.drop('Fraud', axis=1)
        y = data['Fraud']

        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)

        xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        xgb.fit(X, y)

        joblib.dump(rf, rf_model_path)
        joblib.dump(xgb, xgb_model_path)

        return "Models retrained and saved successfully!"
    except Exception as e:
        return f"Error during retraining: {e}"


# ---------- Web Home ----------
@app.route('/')
def home():
    return render_template('index.html')


# ---------- HTML Form Prediction ----------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form
        herb_type = data['herb_type']
        quality_score = float(data['quality_score'])
        moisture_level = float(data['moisture_level'])
        stock_before = float(data['stock_before'])
        stock_after = float(data['stock_after'])
        amount = float(data['amount'])

        le = LabelEncoder()
        le.fit([herb_type])
        herb_encoded = le.transform([herb_type])[0]

        input_data = pd.DataFrame([[
            herb_encoded, quality_score, moisture_level,
            stock_before, stock_after, amount
        ]], columns=['herb_type', 'quality_score', 'moisture_level', 'stock_before', 'stock_after', 'amount'])

        if rf_model is None or xgb_model is None:
            return render_template('result.html', result="No model found. Please retrain first.")

        rf_pred = rf_model.predict(input_data)[0]
        xgb_pred = xgb_model.predict(input_data)[0]

        final_result = "Fraud Detected" if (rf_pred == 1 or xgb_pred == 1) else "Safe Transaction"
        return render_template('result.html', result=final_result)

    except Exception as e:
        return render_template('result.html', result=f"Error: {e}")


# ---------- API Endpoint for Team Use ----------
@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        json_data = request.get_json()
        herb_type = json_data['herb_type']
        quality_score = float(json_data['quality_score'])
        moisture_level = float(json_data['moisture_level'])
        stock_before = float(json_data['stock_before'])
        stock_after = float(json_data['stock_after'])
        amount = float(json_data['amount'])

        le = LabelEncoder()
        le.fit([herb_type])
        herb_encoded = le.transform([herb_type])[0]

        input_data = pd.DataFrame([[
            herb_encoded, quality_score, moisture_level,
            stock_before, stock_after, amount
        ]], columns=['herb_type', 'quality_score', 'moisture_level', 'stock_before', 'stock_after', 'amount'])

        rf_pred = rf_model.predict(input_data)[0]
        xgb_pred = xgb_model.predict(input_data)[0]

        final_result = "Fraud" if (rf_pred == 1 or xgb_pred == 1) else "Safe"

        return jsonify({
            "status": "success",
            "prediction": final_result,
            "rf_prediction": int(rf_pred),
            "xgb_prediction": int(xgb_pred)
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


# ---------- Retrain ----------
@app.route('/retrain', methods=['POST'])
def retrain():
    try:
        if 'dataset' not in request.files:
            return jsonify({'status': 'error', 'message': 'No file uploaded'})

        file = request.files['dataset']
        df = pd.read_csv(file)

        message = train_and_save_models(df)
        return jsonify({'status': 'success', 'message': message})
    except Exception as e:
        return jsonify({'status': 'error', 'message': f"Error during retraining: {e}"})


if __name__ == '__main__':
    app.run(debug=True)

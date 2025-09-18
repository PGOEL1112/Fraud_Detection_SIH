from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import os

app = Flask(__name__)

# Model file paths
RF_MODEL_PATH = "fraud_model_rf.pkl"
XGB_MODEL_PATH = "fraud_model_xgb.pkl"

# Load models if exist
rf_model = joblib.load(RF_MODEL_PATH) if os.path.exists(RF_MODEL_PATH) else None
xgb_model = joblib.load(XGB_MODEL_PATH) if os.path.exists(XGB_MODEL_PATH) else None

# ---------- Train and save models ----------
def train_and_save_models(df):
    try:
        data = df.copy()

        if 'herb_type' in data.columns:
            le = LabelEncoder()
            data['herb_type'] = le.fit_transform(data['herb_type'])

        if 'Fraud' not in data.columns:
            raise ValueError("CSV must have 'Fraud' column")

        X = data.drop('Fraud', axis=1)
        y = data['Fraud']

        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)

        xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        xgb.fit(X, y)

        joblib.dump(rf, RF_MODEL_PATH)
        joblib.dump(xgb, XGB_MODEL_PATH)

        return "Models retrained and saved successfully!"
    except Exception as e:
        return f"Error during retraining: {e}"

# ---------- Routes ----------
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/retrain_page')
def retrain_page():
    return render_template('retrain.html')

@app.route('/predict', methods=['POST'])
def predict():
    global rf_model, xgb_model
    try:
        data = request.form
        herb_type = data['herb_type'].strip()
        quality_score = float(data['quality_score'])
        moisture_level = float(data['moisture_level'])
        stock_before = float(data['stock_before'])
        stock_after = float(data['stock_after'])
        amount = float(data['amount'])

        if rf_model is None or xgb_model is None:
            return jsonify({'result': "No model found. Please retrain first."})

        # ---------- Manual Fraud Rules ----------
        def check_rules(quality_score, moisture_level, stock_before, stock_after, amount):
            if quality_score > 10:
                return "Fraud Detected (Quality Score too high)"
            if moisture_level > 10:
                return "Fraud Detected (Moisture too high)"
            if stock_after > stock_before:
                return "Fraud Detected (Stock inconsistency)"
            if amount > 100000:
                return "Fraud Detected (Suspicious Amount)"
            return None

        rule_result = check_rules(quality_score, moisture_level, stock_before, stock_after, amount)
        if rule_result:
            return jsonify({'result': rule_result})

        # ---------- Handle herb_type ----------
        try:
            le = LabelEncoder()
            le.fit(rf_model.feature_names_in_[:1])  # first column is herb_type
            herb_encoded = le.transform([herb_type])[0]
        except ValueError:
            return jsonify({'result': "Fraud Detected (Unknown Herb Type)"})

        input_data = pd.DataFrame([[herb_encoded, quality_score, moisture_level, stock_before, stock_after, amount]],
                                  columns=['herb_type', 'quality_score', 'moisture_level', 'stock_before', 'stock_after', 'amount'])

        rf_pred = rf_model.predict(input_data)[0]
        xgb_pred = xgb_model.predict(input_data)[0]

        final_result = "Fraud Detected" if (rf_pred == 1 or xgb_pred == 1) else "Safe Transaction"

        return jsonify({'result': final_result})

    except Exception as e:
        return jsonify({'result': f"Error: {e}"})

@app.route('/retrain', methods=['POST'])
def retrain():
    global rf_model, xgb_model
    try:
        if 'dataset' not in request.files:
            return jsonify({'status': 'error', 'message': 'No file uploaded'})

        file = request.files['dataset']
        df = pd.read_csv(file)

        message = train_and_save_models(df)

        # Reload models after retraining
        rf_model = joblib.load(RF_MODEL_PATH)
        xgb_model = joblib.load(XGB_MODEL_PATH)

        return jsonify({'status': 'success', 'message': message})

    except Exception as e:
        return jsonify({'status': 'error', 'message': f"Error during retraining: {e}"})


if __name__ == '__main__':
    app.run(debug=True)

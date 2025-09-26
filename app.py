from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import os

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['DEBUG'] = True

# Paths
rf_model_path = "fraud_model_rf.pkl"
xgb_model_path = "fraud_model_xgb.pkl"
csv_dataset_path = "herbal_transactions_1000.csv"  # auto-retrain if exists

rf_model = None
xgb_model = None

# ---------- Helper Function ----------
def train_and_save_models(df):
    global rf_model, xgb_model
    try:
        data = df.copy()
        if 'herb_type' in data.columns:
            le = LabelEncoder()
            data['herb_type'] = le.fit_transform(data['herb_type'])
        if 'Fraud' not in data.columns:
            raise ValueError("'Fraud' column not found in dataset.")

        X = data.drop('Fraud', axis=1)
        y = data['Fraud']

        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)

        xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        xgb.fit(X, y)

        joblib.dump(rf, rf_model_path)
        joblib.dump(xgb, xgb_model_path)

        rf_model = rf
        xgb_model = xgb

        return "✅ Models retrained and saved successfully!"
    except Exception as e:
        return f"❌ Error during retraining: {e}"

# ---------- Auto retrain ----------
if os.path.exists(csv_dataset_path):
    try:
        df = pd.read_csv(csv_dataset_path)
        print(train_and_save_models(df))
    except Exception as e:
        print(f"❌ Auto retrain failed: {e}")
else:
    print("⚠ CSV dataset not found. Upload manually via /retrain_page.")

# ---------- Routes ----------
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/retrain_page')
def retrain_page():
    return render_template('retrain.html')

# ---------- Prediction ----------
# Load valid herbs from training CSV
valid_herbs = pd.read_csv(csv_dataset_path)['herb_type'].unique().tolist() if os.path.exists(csv_dataset_path) else []

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        json_data = request.get_json()

        # ---------- Read and clean inputs ----------
        herb_type = str(json_data.get('herb_type', '')).strip()
        quality_score = float(json_data.get('quality_score', 0))
        moisture_level = float(json_data.get('moisture_level', 0))
        stock_before = float(json_data.get('stock_before', 0))
        stock_after = float(json_data.get('stock_after', 0))
        amount = float(json_data.get('amount', 0))

        # ---------- Input validation ----------
        errors = []
        valid_herbs = ['Amla','Ashwagandha','Tulsi','Neem','Giloy','Triphala','Shatavari','Brahmi']

        if herb_type not in valid_herbs:
            errors.append(f"Invalid herb type. Choose from {valid_herbs}")
        if quality_score < 0:
            errors.append("Quality score cannot be negative")
        if moisture_level < 0:
            errors.append("Moisture level cannot be negative")
        if stock_before < 0 or stock_after < 0:
            errors.append("Stock values cannot be negative")
        if amount < 0:
            errors.append("Transaction amount cannot be negative")

        if errors:
            return jsonify({"status":"error", "result":"❌ Invalid input", "details": errors}), 400

        # ---------- Encode herb correctly ----------
        le = LabelEncoder()
        le.fit(valid_herbs)
        herb_encoded = le.transform([herb_type])[0]

        input_data = pd.DataFrame([[herb_encoded, quality_score, moisture_level,
                                   stock_before, stock_after, amount]],
                                  columns=['herb_type','quality_score','moisture_level',
                                           'stock_before','stock_after','amount'])

        # ---------- Check obvious safe ranges ----------
        # If none of the rule thresholds are crossed, mark as Safe immediately
        if quality_score <= 10 and (stock_before - stock_after) <= 5 and amount <= 15000:
            return jsonify({
                "status":"success",
                "result": "Safe",
                "reasons": [],
                "rf_prediction": 0,
                "xgb_prediction": 0
            })

        # ---------- ML Predictions ----------
        rf_pred = rf_model.predict(input_data)[0]
        xgb_pred = xgb_model.predict(input_data)[0]

        # ---------- Determine final result and reasons ----------
        final_result = "Safe"
        reasons = []

        if rf_pred == 1 or xgb_pred == 1:
            final_result = "Fraud"
            if quality_score > 10:
                reasons.append("High Quality Score")
            if (stock_before - stock_after) > 5:
                reasons.append("Large Stock Difference")
            if amount > 15000:
                reasons.append("High Transaction Amount")

        return jsonify({
            "status":"success",
            "result": final_result,
            "reasons": reasons,
            "rf_prediction": int(rf_pred),
            "xgb_prediction": int(xgb_pred)
        })

    except Exception as e:
        return jsonify({"status":"error", "result": f"❌ Error: {str(e)}"}), 400

# ---------- Retrain ----------
@app.route('/retrain', methods=['POST'])
def retrain():
    try:
        if 'dataset' not in request.files:
            return jsonify({"status": "error", "message": "❌ No file uploaded"}), 400

        file = request.files['dataset']
        df = pd.read_csv(file)

        message = train_and_save_models(df)

        return jsonify({"status": "success", "message": message})

    except Exception as e:
        return jsonify({"status": "error", "message": f"❌ Error during retraining: {e}"}), 400

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

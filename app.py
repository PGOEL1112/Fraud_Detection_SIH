import os
from flask import Flask, request, jsonify
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Model paths
RF_MODEL_PATH = "fraud_model_rf.pkl"
XGB_MODEL_PATH = "fraud_model_xgb.pkl"
ENCODER_PATH = "label_encoder.pkl"

# Load models if they exist
rf_model = joblib.load(RF_MODEL_PATH) if os.path.exists(RF_MODEL_PATH) else None
xgb_model = joblib.load(XGB_MODEL_PATH) if os.path.exists(XGB_MODEL_PATH) else None
le = joblib.load(ENCODER_PATH) if os.path.exists(ENCODER_PATH) else None

# ---------- Train and Save ----------
def train_and_save_models(df):
    global rf_model, xgb_model, le
    try:
        if "herb_type" not in df.columns or "Fraud" not in df.columns:
            return "Dataset must contain 'herb_type' and 'Fraud' columns"

        df_copy = df.copy()

        # Encode herb_type
        le = LabelEncoder()
        df_copy["herb_type"] = le.fit_transform(df_copy["herb_type"])
        joblib.dump(le, ENCODER_PATH)

        X = df_copy.drop("Fraud", axis=1)
        y = df_copy["Fraud"]

        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X, y)

        xgb_model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
        xgb_model.fit(X, y)

        joblib.dump(rf_model, RF_MODEL_PATH)
        joblib.dump(xgb_model, XGB_MODEL_PATH)

        return "Models trained and saved successfully!"
    except Exception as e:
        return f"Error during training: {e}"


# ---------- API ----------
@app.route("/")
def home():
    return jsonify({"message": "Fraud Detection API is running!"})


@app.route("/predict", methods=["POST"])
def predict():
    global rf_model, xgb_model, le
    if rf_model is None or xgb_model is None or le is None:
        return jsonify({"result": "Models not trained yet. Call /retrain first."}), 400

    try:
        data = request.get_json()
        required = ["herb_type", "quality_score", "moisture_level", "stock_before", "stock_after", "amount"]
        if not all(k in data for k in required):
            return jsonify({"result": "Missing fields"}), 400

        # Manual fraud rules
        if float(data["quality_score"]) > 10:
            return jsonify({"result": "Fraud Detected (Quality Score too high)"})
        if float(data["moisture_level"]) > 10:
            return jsonify({"result": "Fraud Detected (Moisture too high)"})
        if float(data["stock_after"]) > float(data["stock_before"]):
            return jsonify({"result": "Fraud Detected (Stock inconsistency)"})
        if float(data["amount"]) > 100000:
            return jsonify({"result": "Fraud Detected (Suspicious Amount)"})

        # Encode herb_type
        try:
            herb_encoded = le.transform([data["herb_type"]])[0]
        except ValueError:
            return jsonify({"result": "Fraud Detected (Unknown Herb Type)"})

        input_df = pd.DataFrame([[herb_encoded, float(data["quality_score"]), float(data["moisture_level"]),
                                  float(data["stock_before"]), float(data["stock_after"]), float(data["amount"])]],
                                columns=["herb_type", "quality_score", "moisture_level", "stock_before", "stock_after", "amount"])

        rf_pred = rf_model.predict(input_df)[0]
        xgb_pred = xgb_model.predict(input_df)[0]

        result = "Fraud Detected" if rf_pred == 1 or xgb_pred == 1 else "Safe Transaction"
        return jsonify({"result": result})

    except Exception as e:
        return jsonify({"result": f"Error: {e}"}), 500


@app.route("/retrain", methods=["POST"])
def retrain():
    try:
        data = request.get_json()
        if "dataset" not in data:
            return jsonify({"status": "error", "message": "No dataset provided"}), 400

        df = pd.DataFrame(data["dataset"])
        message = train_and_save_models(df)
        return jsonify({"status": "success", "message": message})

    except Exception as e:
        return jsonify({"status": "error", "message": f"Error during retraining: {e}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

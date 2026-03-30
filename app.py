from flask import Flask, render_template, request, jsonify, send_file
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
import pandas as pd
import joblib
import io
import os
from pymongo import MongoClient
from dotenv import load_dotenv
load_dotenv()

# ✅ SAFE MongoDB connection (ENV VARIABLE)
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client["fraud_db"]
collection = db["transactions"]


app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['DEBUG'] = True

# Paths
rf_model_path = "models/fraud_model_rf.pkl"
xgb_model_path = "models/fraud_model_xgb.pkl"
encoder_path = "models/label_encoder.pkl"
csv_dataset_path = "herbal_transactions_1000.csv"  # auto-retrain if exists

rf_model = None
xgb_model = None

# ---------- Helper Function ----------
def train_and_save_models(df):
    global rf_model, xgb_model
    try:
        os.makedirs("models", exist_ok=True)
        data = df.copy()

        if 'Fraud' not in data.columns:
            raise ValueError("'Fraud' column not found in dataset.")

        le = LabelEncoder()
        df['herb_type'] = le.fit_transform(df['herb_type'])
        joblib.dump(le, encoder_path)

        X = df.drop('Fraud', axis=1)
        y = df['Fraud']

        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X, y)

        xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        xgb_model.fit(X, y)

        joblib.dump(rf_model, rf_model_path)
        joblib.dump(xgb_model, xgb_model_path)

        return "✅ Models retrained and saved successfully!"
    except Exception as e:
        return f"❌ Error during retraining: {e}"

# ---------- Auto retrain ----------
if os.path.exists(rf_model_path) and os.path.exists(xgb_model_path) and os.path.exists(encoder_path):
    try:
        rf_model = joblib.load(rf_model_path)
        xgb_model = joblib.load(xgb_model_path)
        print("✅ Models loaded successfully!")

    except Exception as e:
        print(f"❌ Loading failed: {e}")
else:
    if os.path.exists(csv_dataset_path):
        df = pd.read_csv(csv_dataset_path)
        print(train_and_save_models(df))
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
@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json()

        # ---------- Read inputs ----------
        user_email = data.get("email", "guest")
        herb_type = str(data.get('herb_type', '')).strip()
        quality_score = float(data.get('quality_score', 0))
        moisture_level = float(data.get('moisture_level', 0))
        stock_before = float(data.get('stock_before', 0))
        stock_after = float(data.get('stock_after', 0))
        amount = float(data.get('amount', 0))

        # ---------- Validation ----------
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
            return jsonify({
                "status": "error",
                "result": "❌ Invalid input",
                "details": errors
            }), 400

        # ---------- Encode ----------
        le = joblib.load(encoder_path)
        herb_encoded = le.transform([herb_type])[0]

        input_data = pd.DataFrame([[herb_encoded, quality_score, moisture_level,
                                   stock_before, stock_after, amount]],
                                  columns=['herb_type','quality_score','moisture_level',
                                           'stock_before','stock_after','amount'])

        # ---------- Default values ----------
        final_result = "Safe"
        reasons = []
        rf_pred = 0
        xgb_pred = 0
        avg_confidence = 0
        risk = "Low"

        # ---------- Rule-based SAFE ----------
        if quality_score <= 10 and (stock_before - stock_after) <= 5 and amount <= 15000:
            final_result = "Safe"

        else:
            # ---------- ML Prediction ----------
            rf_pred = rf_model.predict(input_data)[0]
            xgb_pred = xgb_model.predict(input_data)[0]

            rf_prob = rf_model.predict_proba(input_data)[0][1]
            xgb_prob = xgb_model.predict_proba(input_data)[0][1]

            avg_confidence = (rf_prob + xgb_prob) / 2

            # Risk level
            if avg_confidence < 0.3:
                risk = "Low"
            elif avg_confidence < 0.7:
                risk = "Medium"
            else:
                risk = "High"

            if rf_pred == 1 or xgb_pred == 1:
                final_result = "Fraud"

                if quality_score > 10:
                    reasons.append("High Quality Score")
                if (stock_before - stock_after) > 5:
                    reasons.append("Large Stock Difference")
                if amount > 15000:
                    reasons.append("High Transaction Amount")

        # ---------- Save to DB (FIXED BUG HERE) ----------
        data_to_save = {
            "email": user_email,  
            "herb_type": herb_type,
            "quality_score": quality_score,
            "moisture_level": moisture_level,
            "stock_before": stock_before,
            "stock_after": stock_after,
            "amount": amount,
            "result": final_result   # ✅ FIXED (was wrong before)
        }

        collection.insert_one(data_to_save)

        # ---------- Response ----------
        return jsonify({
            "status": "success",
            "result": final_result,
            "confidence": round(avg_confidence * 100, 2),
            "risk_level": risk,
            "reasons": reasons,
            "rf_prediction": int(rf_pred),
            "xgb_prediction": int(xgb_pred)
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "result": f"❌ Error: {str(e)}"
        }), 400
    

@app.route('/api/my_data', methods=['POST'])
def get_user_data():
    try:
        email = request.json.get("email")

        data = list(collection.find({"email": email}, {"_id":0}))

        return jsonify({
            "status": "success",
            "data": data
        })

    except Exception as e:
        return jsonify({"status":"error","message":str(e)})
    
    
@app.route('/api/bulk_predict', methods=['POST'])
def bulk_predict():
    try:
        file = request.files['file']
        df = pd.read_csv(file)

        le = joblib.load(encoder_path)

        # encode herb
        df['herb_type'] = le.transform(df['herb_type'])

        X = df[['herb_type','quality_score','moisture_level',
                'stock_before','stock_after','amount']]

        rf_preds = rf_model.predict(X)
        xgb_preds = xgb_model.predict(X)

        results = []
        for i in range(len(df)):
            result = "Fraud" if (rf_preds[i] == 1 or xgb_preds[i] == 1) else "Safe"
            results.append(result)

        df['Prediction'] = results

        fraud_count = results.count("Fraud")
        safe_count = results.count("Safe")

        return jsonify({
            "status": "success",
            "fraud": fraud_count,
            "safe": safe_count,
            "data": df.to_dict(orient='records')
        })

    except Exception as e:
        return jsonify({"status":"error","message":str(e)},400)



@app.route('/download_report')
def download_report():
    try:
        df = pd.read_csv(csv_dataset_path)
        buffer = io.BytesIO()
        df.to_csv(buffer, index=False)
        buffer.seek(0)

        return send_file(buffer,
                         mimetype='text/csv',
                         as_attachment=True,
                         download_name='report.csv')

    except Exception as e:
        return str(e)
    
@app.route('/download_pdf')
def download_pdf():
    try:
        df = pd.read_csv(csv_dataset_path)

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer)

        styles = getSampleStyleSheet()
        elements = []

        elements.append(Paragraph("Fraud Detection Report", styles['Title']))

        data = [df.columns.tolist()] + df.values.tolist()

        table = Table(data[:30])  # first 30 rows

        table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.grey),
            ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),
            ('GRID', (0,0), (-1,-1), 1, colors.black)
        ]))

        elements.append(table)

        doc.build(elements)
        buffer.seek(0)

        return send_file(buffer,
                         mimetype='application/pdf',
                         as_attachment=True,
                         download_name='fraud_report.pdf')

    except Exception as e:
        return str(e)
    
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

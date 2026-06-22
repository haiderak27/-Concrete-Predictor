# app_flask.py - FLASK API VERSION ONLY
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import os

print("=" * 60)
print("🌐 CONCRETE STRENGTH API")
print("=" * 60)

app = Flask(__name__)
CORS(app)

print("\n📂 Loading model artifacts...")

try:
    model = joblib.load('models/concrete_strength_model.pkl')
    scaler = joblib.load('models/concrete_scaler.pkl')
    print("   ✅ Model loaded successfully!")
except FileNotFoundError:
    print("   ❌ ERROR: Model files not found!")
    exit(1)

@app.route('/')
def home():
    return jsonify({
        'message': '🏗️ Concrete Strength Predictor API',
        'status': 'online'
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        features = np.array([[
            data['cement'],
            data['blast_furnace_slag'],
            data['fly_ash'],
            data['water'],
            data['superplasticizer'],
            data['coarse_aggregate'],
            data['fine_aggregate'],
            data['age']
        ]])
        
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        
        return jsonify({
            'strength': round(float(prediction), 2),
            'status': 'success'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

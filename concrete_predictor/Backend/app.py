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

# Create Flask app
app = Flask(__name__)
CORS(app)  # Allow frontend to talk to API

# Load the trained model and scaler
print("\n📂 Loading model artifacts...")

try:
    model = joblib.load('../models/concrete_strength_model.pkl')
    scaler = joblib.load('../models/concrete_scaler.pkl')
    metadata = joblib.load('../models/model_metadata.pkl')
    print("   ✅ Model loaded successfully!")
    print(f"   R² Score: {metadata['r2_score']:.4f}")
    print(f"   MAE: {metadata['mae']:.2f} MPa")
except FileNotFoundError as e:
    print(f"   ❌ ERROR: {e}")
    print("   Please run train_model.py first!")
    exit(1)

# Feature columns (must match training)
FEATURE_COLS = [
    'cement', 'blast_furnace_slag', 'fly_ash', 'water', 
    'superplasticizer', 'coarse_aggregate', 'fine_aggregate', 'age'
]

# Validation rules
VALIDATION_RULES = {
    'cement': {'min': 100, 'max': 600},
    'blast_furnace_slag': {'min': 0, 'max': 450},
    'fly_ash': {'min': 0, 'max': 600},
    'water': {'min': 100, 'max': 250},
    'superplasticizer': {'min': 0, 'max': 30},
    'coarse_aggregate': {'min': 600, 'max': 1500},
    'fine_aggregate': {'min': 500, 'max': 1200},
    'age': {'min': 1, 'max': 365}
}

@app.route('/')
def home():
    """API home page"""
    return jsonify({
        'message': '🏗️ Concrete Strength Predictor API',
        'status': 'online',
        'model_version': 'v1.0',
        'endpoints': {
            '/predict': 'POST - Predict concrete strength',
            '/health': 'GET - Check API health',
            '/metadata': 'GET - View model info',
            '/': 'GET - This page'
        }
    })

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': True,
        'timestamp': datetime.utcnow().isoformat() + 'Z'
    })

@app.route('/metadata')
def get_metadata():
    """Get model performance metrics"""
    return jsonify(metadata)

@app.route('/predict', methods=['POST'])
def predict():
    """Predict concrete compressive strength"""
    try:
        # Get JSON data
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Check for missing features
        missing = [f for f in FEATURE_COLS if f not in data]
        if missing:
            return jsonify({
                'error': f'Missing features: {missing}',
                'status': 'error'
            }), 400
        
        # Extract and validate values
        input_values = {}
        for feature in FEATURE_COLS:
            try:
                value = float(data[feature])
            except (TypeError, ValueError):
                return jsonify({
                    'error': f'Invalid value for {feature}: must be a number',
                    'status': 'error'
                }), 400
            
            # Domain validation
            rule = VALIDATION_RULES[feature]
            if value < rule['min'] or value > rule['max']:
                return jsonify({
                    'error': f'{feature} must be between {rule["min"]} and {rule["max"]} (got {value})',
                    'status': 'error'
                }), 400
            
            input_values[feature] = value
        
        # Check water/cement ratio
        w_c_ratio = input_values['water'] / input_values['cement']
        if w_c_ratio < 0.25:
            return jsonify({
                'error': f'Water/Cement ratio ({w_c_ratio:.3f}) is too low. Minimum is 0.25.',
                'status': 'error'
            }), 400
        if w_c_ratio > 0.8:
            return jsonify({
                'error': f'Water/Cement ratio ({w_c_ratio:.3f}) is too high. Maximum is 0.8.',
                'status': 'error'
            }), 400
        
        # Prepare input
        input_df = pd.DataFrame([input_values])[FEATURE_COLS]
        input_scaled = scaler.transform(input_df)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        prediction = max(prediction, 0)  # Strength can't be negative
        prediction = min(prediction, 150)  # Cap at realistic max
        
        # Return result
        return jsonify({
            'strength': round(float(prediction), 2),
            'status': 'success',
            'model_version': 'v1.0',
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'input': {k: round(v, 2) for k, v in input_values.items()},
            'w_c_ratio': round(w_c_ratio, 3)
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Server error: {str(e)}',
            'status': 'error'
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"\n🚀 Starting server at http://127.0.0.1:{port}")
    print("   Press CTRL+C to stop the server")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=port)
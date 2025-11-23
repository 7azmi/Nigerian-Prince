#!/usr/bin/env python3
"""
Flask web server for Nigerian Prince Fraud Email Detection
Serves static HTML page and provides API endpoint for predictions
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pickle
import os

app = Flask(__name__)
CORS(app)

# Load model and vectorizer on startup
print("Loading model...")
try:
    with open('fraud_detection_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    
    print("✓ Model loaded successfully!")
except FileNotFoundError:
    print("Error: Model files not found. Please run train_model.py first.")
    model = None
    vectorizer = None


@app.route('/')
def index():
    """Serve the main HTML page"""
    return send_from_directory('.', 'index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for email fraud prediction"""
    if model is None or vectorizer is None:
        return jsonify({
            'error': 'Model not loaded. Please train the model first.'
        }), 500
    
    try:
        data = request.get_json()
        email_text = data.get('email_text', '')
        
        if not email_text:
            return jsonify({
                'error': 'No email text provided'
            }), 400
        
        # Transform and predict
        email_features = vectorizer.transform([email_text])
        prediction = model.predict(email_features)[0]
        probability = model.predict_proba(email_features)[0]
        
        result = {
            'prediction': int(prediction),
            'label': 'FRAUD' if prediction == 1 else 'LEGITIMATE',
            'fraud_probability': float(probability[1]),
            'legitimate_probability': float(probability[0]),
            'confidence': float(max(probability))
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            'error': f'Prediction failed: {str(e)}'
        }), 500


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None and vectorizer is not None
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)

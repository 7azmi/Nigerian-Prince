#!/usr/bin/env python3
"""
Prediction script for Nigerian Prince Fraud Email Detection
Can be used standalone or called from web interface
"""

import pickle
import json
import sys


def load_model_artifacts():
    """Load the trained model and vectorizer"""
    try:
        with open('fraud_detection_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        with open('tfidf_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        
        return model, vectorizer
    except FileNotFoundError as e:
        print(f"Error: Model files not found. Please run train_model.py first.")
        print(f"Missing file: {e.filename}")
        sys.exit(1)


def predict_email(email_text, model=None, vectorizer=None):
    """
    Predict if an email is fraudulent or legitimate
    
    Args:
        email_text: The email text to analyze
        model: Pre-loaded model (optional)
        vectorizer: Pre-loaded vectorizer (optional)
    
    Returns:
        dict with prediction, probability, and label
    """
    # Load model if not provided
    if model is None or vectorizer is None:
        model, vectorizer = load_model_artifacts()
    
    # Transform text to features
    email_features = vectorizer.transform([email_text])
    
    # Make prediction
    prediction = model.predict(email_features)[0]
    probability = model.predict_proba(email_features)[0]
    
    # Get probability for fraud class (class 1)
    fraud_probability = probability[1]
    
    result = {
        'prediction': int(prediction),
        'label': 'FRAUD' if prediction == 1 else 'LEGITIMATE',
        'fraud_probability': float(fraud_probability),
        'legitimate_probability': float(probability[0]),
        'confidence': float(max(probability))
    }
    
    return result


def main():
    """Command-line interface for prediction"""
    if len(sys.argv) < 2:
        print("Usage: python predict.py <email_text>")
        print("   or: python predict.py --interactive")
        sys.exit(1)
    
    # Load model once
    model, vectorizer = load_model_artifacts()
    
    if sys.argv[1] == '--interactive':
        print("="*60)
        print("Nigerian Prince Fraud Email Detection - Interactive Mode")
        print("="*60)
        print("Enter email text (press Ctrl+D or Ctrl+Z when done):")
        print()
        
        try:
            email_text = sys.stdin.read()
        except KeyboardInterrupt:
            print("\nExiting...")
            sys.exit(0)
    else:
        email_text = ' '.join(sys.argv[1:])
    
    # Make prediction
    result = predict_email(email_text, model, vectorizer)
    
    # Display result
    print("\n" + "="*60)
    print("PREDICTION RESULT")
    print("="*60)
    print(f"\nEmail Classification: {result['label']}")
    print(f"Confidence: {result['confidence']*100:.2f}%")
    print(f"\nProbabilities:")
    print(f"  Fraud:      {result['fraud_probability']*100:.2f}%")
    print(f"  Legitimate: {result['legitimate_probability']*100:.2f}%")
    
    if result['label'] == 'FRAUD':
        print("\n⚠️  WARNING: This email appears to be FRAUDULENT!")
        print("   Do not respond or provide any personal information.")
    else:
        print("\n✓ This email appears to be LEGITIMATE.")
    
    print("="*60)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Nigerian Prince Fraud Email Detection - Training Script
This script trains a Random Forest classifier to detect fraudulent emails
with proper data splitting and comprehensive evaluation metrics.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
import pickle
import json
from datetime import datetime


def load_data(file_path='fraud_email_.csv'):
    """Load the email dataset"""
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip')
    print(f"Total samples: {len(df)}")
    print(f"Fraud emails (1): {(df['Class'] == 1).sum()}")
    print(f"Legitimate emails (0): {(df['Class'] == 0).sum()}")
    return df


def split_data(X, y, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, random_state=42):
    """
    Split data into training, validation, and testing sets
    Default: 70% train, 20% validation, 10% test
    """
    # First split: separate test set (10%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=random_state, stratify=y
    )
    
    # Second split: separate train and validation from remaining 90%
    # validation should be 20% of total = 20/90 of the temp set
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio_adjusted, 
        random_state=random_state, stratify=y_temp
    )
    
    print(f"\nData split:")
    print(f"  Training set: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  Validation set: {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
    print(f"  Test set: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def create_features(X_train, X_val, X_test, max_features=5000):
    """Create TF-IDF features from text data"""
    print("\nCreating TF-IDF features...")
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=2,
        max_df=0.95,
        stop_words='english',
        ngram_range=(1, 2)
    )
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)
    X_test_tfidf = vectorizer.transform(X_test)
    
    print(f"Feature matrix shape: {X_train_tfidf.shape}")
    
    return X_train_tfidf, X_val_tfidf, X_test_tfidf, vectorizer


def train_random_forest(X_train, y_train, X_val, y_val):
    """Train Random Forest classifier"""
    print("\nTraining Random Forest classifier...")
    
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=50,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    rf_model.fit(X_train, y_train)
    
    # Evaluate on validation set
    val_predictions = rf_model.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_predictions)
    
    print(f"\nValidation accuracy: {val_accuracy:.4f}")
    
    return rf_model


def evaluate_model(model, X_test, y_test):
    """
    Comprehensive model evaluation with confusion matrix and metrics
    """
    print("\n" + "="*60)
    print("MODEL EVALUATION ON TEST SET (UNSEEN DATA)")
    print("="*60)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print("\n📊 CONFUSION MATRIX:")
    print("-" * 40)
    print(f"                 Predicted")
    print(f"                 0         1")
    print(f"Actual    0    {tn:5d}    {fp:5d}")
    print(f"          1    {fn:5d}    {tp:5d}")
    print("-" * 40)
    
    print("\n📈 CONFUSION MATRIX COMPONENTS:")
    print(f"  True Negatives (TN):  {tn:5d}  ✓ Correctly identified legitimate emails")
    print(f"  False Positives (FP): {fp:5d}  ✗ Legitimate emails marked as fraud")
    print(f"  False Negatives (FN): {fn:5d}  ✗ Fraud emails marked as legitimate")
    print(f"  True Positives (TP):  {tp:5d}  ✓ Correctly identified fraud emails")
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print("\n📋 PERFORMANCE METRICS TABLE:")
    print("-" * 40)
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"  Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"  F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
    print("-" * 40)
    
    print("\n📊 DETAILED CLASSIFICATION REPORT:")
    print(classification_report(y_test, y_pred, 
                              target_names=['Legitimate (0)', 'Fraud (1)']))
    
    # Diagonal analysis
    diagonal_sum = tn + tp
    total = tn + fp + fn + tp
    diagonal_percentage = (diagonal_sum / total) * 100
    
    print(f"\n✨ DIAGONAL ANALYSIS:")
    print(f"  Diagonal sum (TN + TP): {diagonal_sum}/{total}")
    print(f"  Diagonal percentage: {diagonal_percentage:.2f}%")
    
    if diagonal_percentage > 90:
        print("  ✓ EXCELLENT: Model performs very well!")
    elif diagonal_percentage > 80:
        print("  ✓ GOOD: Model performs well!")
    elif diagonal_percentage > 70:
        print("  ⚠ FAIR: Model performance is acceptable")
    else:
        print("  ✗ POOR: Model needs improvement")
    
    print("\n" + "="*60)
    
    # Return metrics for saving
    return {
        'confusion_matrix': {
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'tp': int(tp)
        },
        'metrics': {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1)
        },
        'diagonal_percentage': float(diagonal_percentage)
    }


def save_model(model, vectorizer, metrics, output_dir='.'):
    """Save trained model, vectorizer, and metrics"""
    print("\nSaving model and artifacts...")
    
    # Save model
    model_path = f"{output_dir}/fraud_detection_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"  ✓ Model saved to {model_path}")
    
    # Save vectorizer
    vectorizer_path = f"{output_dir}/tfidf_vectorizer.pkl"
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)
    print(f"  ✓ Vectorizer saved to {vectorizer_path}")
    
    # Save metrics
    metrics['timestamp'] = datetime.now().isoformat()
    metrics_path = f"{output_dir}/model_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"  ✓ Metrics saved to {metrics_path}")


def main():
    """Main training pipeline"""
    print("="*60)
    print("NIGERIAN PRINCE FRAUD EMAIL DETECTION - TRAINING")
    print("="*60)
    
    # Load data
    df = load_data()
    
    # Prepare features and labels
    X = df['Text'].fillna('')
    y = df['Class']
    
    # Split data: 70% train, 20% validation, 10% test
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    
    # Create TF-IDF features
    X_train_tfidf, X_val_tfidf, X_test_tfidf, vectorizer = create_features(
        X_train, X_val, X_test
    )
    
    # Train Random Forest model
    model = train_random_forest(X_train_tfidf, y_train, X_val_tfidf, y_val)
    
    # Evaluate on test set (unseen data)
    metrics = evaluate_model(model, X_test_tfidf, y_test)
    
    # Save model and artifacts
    save_model(model, vectorizer, metrics)
    
    print("\n✅ Training completed successfully!")
    print("\nFiles created:")
    print("  - fraud_detection_model.pkl (trained model)")
    print("  - tfidf_vectorizer.pkl (text vectorizer)")
    print("  - model_metrics.json (evaluation metrics)")


if __name__ == '__main__':
    main()

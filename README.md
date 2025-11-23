# 🛡️ Nigerian Prince Fraud Email Detection

A machine learning project that detects fraudulent emails (Nigerian Prince scams) using a Random Forest classifier with comprehensive evaluation metrics and an interactive web demo.

## 📊 Dataset

The project uses a dataset of **11,930 emails** with binary classification:
- **5,187 fraudulent emails** (Class 1)
- **6,742 legitimate emails** (Class 0)

## 🎯 Features

- **Random Forest Classifier** with TF-IDF text vectorization
- **Proper data splitting**: 70% training, 20% validation, 10% testing
- **Comprehensive evaluation metrics**:
  - Confusion Matrix (TP, FP, TN, FN)
  - Accuracy, Precision, Recall, F1-Score
  - Diagonal analysis for model performance
- **Interactive web demo** with static HTML page
- **Command-line prediction** interface
- **REST API** for predictions

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python train_model.py
```

This will:
- Split data into 70% train, 20% validation, 10% test
- Train a Random Forest classifier
- Generate confusion matrix and metrics
- Save the trained model and vectorizer

**Output files:**
- `fraud_detection_model.pkl` - Trained model
- `tfidf_vectorizer.pkl` - Text vectorizer
- `model_metrics.json` - Evaluation metrics

### 3. Run the Web Demo

```bash
python app.py
```

Then open your browser to: `http://localhost:5000`

### 4. Command-Line Predictions

```bash
# Direct prediction
python predict.py "Your email text here"

# Interactive mode
python predict.py --interactive
```

## 📈 Model Performance

After training, the model provides detailed evaluation metrics:

### Confusion Matrix
```
                 Predicted
                 0         1
Actual    0    [TN]     [FP]
          1    [FN]     [TP]
```

Where:
- **TN (True Negatives)**: Correctly identified legitimate emails
- **FP (False Positives)**: Legitimate emails incorrectly marked as fraud
- **FN (False Negatives)**: Fraud emails incorrectly marked as legitimate
- **TP (True Positives)**: Correctly identified fraud emails

### Performance Metrics
- **Accuracy**: Overall correctness
- **Precision**: Accuracy of fraud predictions
- **Recall**: Ability to find all fraud emails
- **F1-Score**: Harmonic mean of precision and recall

### Diagonal Analysis
The sum of TN + TP divided by total predictions indicates overall model performance:
- **>90%**: Excellent performance
- **>80%**: Good performance
- **>70%**: Fair performance
- **<70%**: Needs improvement

## 🌐 Web Demo Features

The interactive web demo (`index.html`) provides:
- Clean, modern UI with gradient design
- Real-time email fraud detection
- Visual confidence indicators
- Probability breakdown (fraud vs. legitimate)
- Example emails to test
- Warning messages for detected fraud
- Responsive design for mobile devices

## 📁 Project Structure

```
Nigerian-Prince/
├── fraud_email_.csv           # Dataset
├── train_model.py            # Training script
├── predict.py                # Prediction CLI
├── app.py                    # Flask web server
├── index.html                # Web demo interface
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── fraud_detection_model.pkl # Trained model (generated)
├── tfidf_vectorizer.pkl      # Vectorizer (generated)
└── model_metrics.json        # Metrics (generated)
```

## 🔬 Technical Details

### Algorithm: Random Forest
- **n_estimators**: 100 trees
- **max_depth**: 50
- **min_samples_split**: 5
- **min_samples_leaf**: 2
- Parallel processing enabled

### Text Vectorization: TF-IDF
- **max_features**: 5000
- **ngram_range**: (1, 2) - unigrams and bigrams
- **min_df**: 2
- **max_df**: 0.95
- English stop words removed

### Data Split Strategy
1. **Test set (10%)**: Separated first for unbiased evaluation
2. **Train/Val split**: Remaining 90% split into 70/20
3. **Stratified sampling**: Maintains class distribution

## 💡 Usage Examples

### Training
```python
python train_model.py
```

### Web Server
```python
python app.py
# Visit http://localhost:5000
```

### Command-Line Prediction
```python
python predict.py "Dear friend, I am a Nigerian prince..."
```

### Programmatic Usage
```python
from predict import predict_email

result = predict_email("Your email text here")
print(f"Label: {result['label']}")
print(f"Confidence: {result['confidence']:.2%}")
```

## 🎓 Educational Notes

### Why Random Forest?
- Robust to overfitting
- Handles high-dimensional data well
- Provides feature importance
- No need for feature scaling
- Works well with text features

### Data Splitting Importance
- **Training (70%)**: Learn patterns
- **Validation (20%)**: Tune hyperparameters
- **Testing (10%)**: Unbiased performance evaluation

### Confusion Matrix Insights
- **High diagonal values** = Good model
- **False Positives**: Legitimate emails blocked (user annoyance)
- **False Negatives**: Fraud emails missed (security risk)

## 🔒 Security Considerations

This model is for **educational purposes**. In production:
- Combine with additional security measures
- Regular retraining with new fraud patterns
- Human review for high-stakes decisions
- Monitor for false positives/negatives

## 📝 License

This project is for educational purposes.

## 🤝 Contributing

Feel free to:
- Test with different algorithms (try Autoencoder as suggested!)
- Improve the web interface
- Add more features
- Enhance the model

## 📚 References

- Dataset: Fraud email classification
- Algorithm: Random Forest (scikit-learn)
- Framework: Flask for web serving
- Visualization: Confusion matrix and metrics

# Usage Guide

## Quick Start Guide

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Train the Model
```bash
python train_model.py
```

This will:
- Load and split the dataset (70/20/10)
- Train a Random Forest classifier
- Display comprehensive metrics
- Save model files:
  - `fraud_detection_model.pkl`
  - `tfidf_vectorizer.pkl`
  - `model_metrics.json`

### Step 3: Use the Model

#### Option A: Web Demo (Recommended)
```bash
python app.py
```
Then open your browser to `http://localhost:5000`

#### Option B: Command Line
```bash
# Direct prediction
python predict.py "Your email text here"

# Interactive mode
python predict.py --interactive
```

## Understanding the Results

### Confusion Matrix
```
                 Predicted
                 0         1
Actual    0     TN        FP
          1     FN        TP
```

- **TN (True Negative)**: Correctly identified legitimate emails ✓
- **FP (False Positive)**: Legitimate emails wrongly marked as fraud ✗
- **FN (False Negative)**: Fraud emails wrongly marked as legitimate ✗
- **TP (True Positive)**: Correctly identified fraud emails ✓

### Diagonal Analysis
The sum of TN + TP (diagonal values) indicates overall accuracy:
- **> 90%**: Excellent performance ✓
- **80-90%**: Good performance
- **70-80%**: Fair performance
- **< 70%**: Needs improvement

## Model Performance

Current model achieves:
- **98.24% Accuracy**
- **99.40% Precision**
- **96.53% Recall**
- **97.95% F1-Score**

This means:
- Only 3 legitimate emails were incorrectly flagged as fraud
- Only 18 fraud emails were missed
- 98.24% of all predictions were correct

## Example Usage

### Testing a Fraud Email
```bash
python predict.py "Dear friend, I am a Nigerian prince and have 25 million dollars to transfer..."
```

Expected output:
```
Email Classification: FRAUD
Confidence: 90.38%
⚠️  WARNING: This email appears to be FRAUDULENT!
```

### Testing a Legitimate Email
```bash
python predict.py "Hi team, meeting tomorrow at 10am to discuss project status."
```

Expected output:
```
Email Classification: LEGITIMATE
Confidence: 87.45%
✓ This email appears to be LEGITIMATE.
```

## Web Interface Features

The HTML demo provides:
1. **Text input area** for email content
2. **Analyze button** to run prediction
3. **Visual results** with:
   - Fraud/Legitimate classification
   - Confidence percentage
   - Probability breakdown
   - Security warnings
4. **Example emails** to test
5. **Responsive design** for mobile/desktop

## Troubleshooting

### Error: "Model files not found"
**Solution**: Run `python train_model.py` first to generate the model files.

### Error: "No module named 'sklearn'"
**Solution**: Install dependencies with `pip install -r requirements.txt`

### Port 5000 already in use
**Solution**: Stop other apps using port 5000, or set a different port:
```bash
export PORT=8080
python app.py
```

## Production Deployment

For production use:
1. Set debug mode to false:
```bash
export FLASK_DEBUG=false
python app.py
```

2. Use a production WSGI server (e.g., gunicorn):
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

3. Consider adding:
   - HTTPS/SSL certificates
   - Rate limiting
   - Authentication
   - Logging and monitoring

## Retraining the Model

To retrain with updated data:
1. Update `fraud_email_.csv` with new emails
2. Run `python train_model.py`
3. Review the new metrics
4. Restart the Flask app

## API Endpoints

### POST /predict
Predict if an email is fraudulent.

**Request:**
```json
{
  "email_text": "Your email content here"
}
```

**Response:**
```json
{
  "prediction": 1,
  "label": "FRAUD",
  "fraud_probability": 0.975,
  "legitimate_probability": 0.025,
  "confidence": 0.975
}
```

### GET /health
Check if the API is running and model is loaded.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

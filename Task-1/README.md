# Task 1 - Network Intrusion Detection using Machine Learning

## Overview

This project implements a **binary classification model** for detecting network intrusions using the **KDD Cup 1999 dataset**. The model distinguishes between normal network traffic and attack traffic, which is critical for Security Operations Center (SOC) and SIEM integration.

**Model Type**: Random Forest Classifier with preprocessing pipeline  
**Dataset**: KDD Cup 1999 (10% subset - 494,021 samples)  
**Key Metrics**: Precision, Recall, F1-Score, ROC AUC

---

##  Project Goals

1. Build a production-ready ML model for network intrusion detection
2. Achieve high recall to minimize missed attacks (false negatives)
3. Maintain good precision to reduce alert fatigue (false positives)
4. Provide model explainability through feature importance analysis
5. Create a reproducible environment and saved model for deployment

---

##  Project Structure

```
Task-1/
│
├── Task1_kdd_rf.ipynb          # Main Jupyter notebook with complete pipeline
├── kdd_rf_pipeline.joblib      # Saved model pipeline (generated after running notebook)
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

##  Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. **Clone or navigate to the project directory**:
   ```powershell
   cd c:\Users\prati\Internship_Assignment-Goklyn\Task-1
   ```

2. **Install required packages**:
   ```powershell
   pip install -r requirements.txt
   ```

3. **Launch Jupyter Notebook**:
   ```powershell
   jupyter notebook
   ```

4. **Open and run** `Task1_kdd_rf.ipynb`

### Running the Notebook

The notebook is structured into 12 sections:

1. **Import Libraries** - Load all required dependencies
2. **Load Dataset** - Fetch KDD Cup 1999 data automatically
3. **Preprocessing** - Clean and prepare data
4. **Build Pipeline** - Create preprocessing + model pipeline
5. **Train-Test Split** - Stratified 80-20 split
6. **Train Model** - Fit Random Forest classifier
7. **Evaluate** - Comprehensive metrics and visualizations
8. **Feature Importance** - Identify key attack indicators
9. **Cross-Validation** - 5-fold CV for robustness
10. **Save Model** - Export pipeline for deployment
11. **Summary** - Results and SOC integration notes
12. **Example Predictions** - Demo on new data

**Execution time**: Approximately 3-5 minutes on a standard laptop

---

## Model Performance

### Expected Results (on test set):

| Metric | Value |
|--------|-------|
| **Accuracy** | ~99.5% |
| **ROC AUC** | ~0.99+ |
| **Precision (Attack class)** | ~99%+ |
| **Recall (Attack class)** | ~99%+ |
| **F1-Score (Attack class)** | ~99%+ |

### Key Performance Indicators for SOC:

- **False Negatives (Missed Attacks)**: Critical metric - model achieves < 1% miss rate
- **False Positives (False Alarms)**: Kept low to prevent alert fatigue
- **Inference Speed**: Real-time capable (< 1ms per prediction)

---

## Dataset Information

**Source**: [KDD Cup 1999 - Network Intrusion Detection](http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html)

**Features**: 41 network traffic attributes including:
- Protocol type (TCP, UDP, ICMP)
- Service type (HTTP, FTP, SMTP, etc.)
- Connection statistics (duration, bytes sent/received)
- Content features (failed logins, compromised conditions)
- Traffic patterns (count, error rates)

**Labels**:
- **Normal**: Legitimate network traffic (Class 0)
- **Attack**: Various attack types including DoS, Probe, R2L, U2R (Class 1)

**Subset Used**: 10% subset (~494K samples) for computational efficiency  
**Class Distribution**: ~20% normal, ~80% attacks

---

##  Model Architecture

### Preprocessing Pipeline:
1. **Categorical Features** (f1, f2, f3): One-Hot Encoding
   - Protocol type, service, flag
2. **Numeric Features** (f4-f40): Standard Scaling
   - Z-score normalization for consistent ranges

### Classifier:
- **Algorithm**: Random Forest
- **Number of Trees**: 200
- **Max Depth**: 20
- **Min Samples Split**: 10
- **Min Samples Leaf**: 4
- **Parallelization**: Multi-core enabled

---

##  Model Deployment

### Saved Artifacts:

1. **kdd_rf_pipeline.joblib** (~150 MB)
   - Complete pipeline (preprocessing + trained model)
   - Ready for production deployment

### Loading and Using the Model:

```python
import joblib
import pandas as pd

# Load the pipeline
pipeline = joblib.load('kdd_rf_pipeline.joblib')

# Prepare new data (same 41 features as training)
new_data = pd.DataFrame({...})  # Your network traffic data

# Make predictions
predictions = pipeline.predict(new_data)
probabilities = pipeline.predict_proba(new_data)[:, 1]

# Generate alerts for attacks
for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
    if pred == 1:
        print(f"Alert! Attack detected with {prob:.2%} confidence")
```

---

##  SIEM Integration Guidelines

### Integration Approach:

1. **Input Format**: 41 network traffic features per connection
2. **Output Format**: Binary classification + probability score
3. **Alert Generation**:
   - Trigger when `prediction == 1`
   - Prioritize by probability score (higher = more confident)
4. **Alert Fields**:
   - Timestamp
   - Source/Destination IPs
   - Attack probability
   - Top contributing features

### Recommended Deployment:

- **Real-time Mode**: Stream network logs through model
- **Batch Mode**: Process historical logs for threat hunting
- **Threshold Tuning**: Adjust probability threshold based on SOC capacity
- **Model Retraining**: Weekly with new labeled data

---

##  Evaluation Visualizations

The notebook generates the following visualizations:

1. **Class Distribution** - Bar chart of normal vs attack samples
2. **Confusion Matrix** - Heatmap showing TP, TN, FP, FN
3. **ROC Curve** - True Positive Rate vs False Positive Rate
4. **Precision-Recall Curve** - Critical for imbalanced datasets
5. **Feature Importance** - Top 20 features driving predictions

---

##  Customization and Extensions

### Hyperparameter Tuning:

Use GridSearchCV or RandomizedSearchCV to optimize:
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'clf__n_estimators': [100, 200, 300],
    'clf__max_depth': [15, 20, 25],
    'clf__min_samples_split': [5, 10, 15]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1')
grid_search.fit(X_train, y_train)
```

### Multiclass Classification:

To detect specific attack types (DoS, Probe, R2L, U2R):
- Remove binary label conversion
- Use original KDD labels
- Adjust metrics for multiclass

### Alternative Models:

- **XGBoost/LightGBM**: Gradient boosting for better accuracy
- **Neural Networks**: Deep learning with Keras/TensorFlow
- **Ensemble**: Combine multiple models for robustness

---

##  Dependencies

Main libraries used:
- **scikit-learn**: ML pipeline and Random Forest
- **pandas**: Data manipulation
- **numpy**: Numerical operations
- **matplotlib/seaborn**: Visualizations
- **joblib**: Model persistence

See `requirements.txt` for complete list with versions.

Results Summary

After running the notebook, you should observe:

- **High accuracy** (~99.5%+) on test data
- **Excellent ROC AUC** (~0.99+) showing strong discriminative power
- **Low false negative rate** (< 1%) - critical for catching attacks
- **Manageable false positive rate** (< 1%) - prevents alert fatigue
- **Clear feature importance** showing which network characteristics matter most

The model is **production-ready** and suitable for integration into SIEM systems for real-time threat detection.


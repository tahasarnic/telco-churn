# Packages
import pandas as pd
import joblib
import mlflow
import mlflow.pyfunc

# Configuration
TRACKING_URI = "mlruns"
MODEL_NAME = "telco_churn_logistic_regression"
PREPROCESSING_PIPELINE_PATH = "data/preprocessing_pipeline.pkl"

# 1. Load production model from MLflow
print("="*60)
print("LOADING PRODUCTION MODEL")
print("="*60)

mlflow.set_tracking_uri(TRACKING_URI)
model_uri = f"models:/{MODEL_NAME}@production"
print(f"Loading model from: {model_uri}")
model = mlflow.pyfunc.load_model(model_uri)
print("✓ Model loaded successfully\n")

# 2. Create 10 sample dummy data (similar to data/data.csv structure)
print("="*60)
print("CREATING SAMPLE DATA")
print("="*60)

sample_data = pd.DataFrame({
    'customerID': ['CUST-001', 'CUST-002', 'CUST-003', 'CUST-004', 'CUST-005',
                   'CUST-006', 'CUST-007', 'CUST-008', 'CUST-009', 'CUST-010'],
    'gender': ['Male', 'Female', 'Male', 'Female', 'Male',
               'Female', 'Male', 'Female', 'Male', 'Female'],
    'SeniorCitizen': [0, 1, 0, 0, 1, 0, 1, 0, 0, 1],
    'Partner': ['Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No'],
    'Dependents': ['No', 'No', 'Yes', 'No', 'Yes', 'No', 'No', 'Yes', 'No', 'Yes'],
    'tenure': [1, 34, 2, 45, 8, 22, 10, 28, 62, 13],
    'PhoneService': ['Yes', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'Yes'],
    'MultipleLines': ['No', 'Yes', 'No', 'No phone service', 'Yes', 'No', 'No phone service', 'Yes', 'No', 'Yes'],
    'InternetService': ['DSL', 'Fiber optic', 'DSL', 'DSL', 'Fiber optic', 'DSL', 'DSL', 'Fiber optic', 'Fiber optic', 'DSL'],
    'OnlineSecurity': ['No', 'No', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'No', 'Yes', 'No'],
    'OnlineBackup': ['Yes', 'No', 'Yes', 'No', 'No', 'Yes', 'No', 'Yes', 'Yes', 'No'],
    'DeviceProtection': ['No', 'Yes', 'No', 'Yes', 'No', 'No', 'Yes', 'Yes', 'No', 'Yes'],
    'TechSupport': ['No', 'No', 'No', 'Yes', 'No', 'Yes', 'Yes', 'No', 'Yes', 'No'],
    'StreamingTV': ['No', 'No', 'No', 'No', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'No'],
    'StreamingMovies': ['No', 'No', 'No', 'No', 'Yes', 'No', 'No', 'Yes', 'Yes', 'Yes'],
    'Contract': ['Month-to-month', 'One year', 'Month-to-month', 'One year', 'Month-to-month',
                 'Two year', 'Month-to-month', 'One year', 'Two year', 'Month-to-month'],
    'PaperlessBilling': ['Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'No', 'Yes'],
    'PaymentMethod': ['Electronic check', 'Mailed check', 'Mailed check', 'Bank transfer (automatic)',
                     'Electronic check', 'Credit card (automatic)', 'Electronic check', 'Bank transfer (automatic)',
                     'Credit card (automatic)', 'Electronic check'],
    'MonthlyCharges': [29.85, 56.95, 53.85, 42.30, 70.70, 89.10, 29.75, 104.80, 84.80, 80.85],
    'TotalCharges': ['29.85', '1889.5', '108.15', '1840.75', '151.65',
                     '1949.4', '346.45', '3046.05', '5377.8', '1346.9']
})

print(f"Created {len(sample_data)} sample customer records")
print("\nSample data preview:")
print(sample_data[['customerID', 'tenure', 'MonthlyCharges', 'Contract']].head())
print()

# 3. Load preprocessing pipeline and preprocess data
print("="*60)
print("PREPROCESSING DATA")
print("="*60)

print(f"Loading preprocessing pipeline from: {PREPROCESSING_PIPELINE_PATH}")
preprocessing_pipeline = joblib.load(PREPROCESSING_PIPELINE_PATH)
print("✓ Pipeline loaded successfully")

# Convert TotalCharges to numeric (as done in training)
sample_data['TotalCharges'] = pd.to_numeric(sample_data['TotalCharges'], errors='coerce')

# Apply preprocessing
preprocessed_data = preprocessing_pipeline.transform(sample_data)
print(f"✓ Data preprocessed successfully")
print(f"Preprocessed shape: {preprocessed_data.shape}\n")

# 4. Make predictions
print("="*60)
print("MAKING PREDICTIONS")
print("="*60)

predictions = model.predict(preprocessed_data)

# Try to get probabilities
try:
    if hasattr(model, '_model_impl') and hasattr(model._model_impl, 'predict_proba'):
        probabilities = model._model_impl.predict_proba(preprocessed_data)[:, 1]
    else:
        probabilities = predictions.astype(float)
except Exception:
    probabilities = predictions.astype(float)

print("✓ Predictions completed\n")

# 5. Save and display predictions
print("="*60)
print("PREDICTION RESULTS")
print("="*60)

# Create results dataframe
results = pd.DataFrame({
    'customerID': sample_data['customerID'],
    'tenure': sample_data['tenure'],
    'MonthlyCharges': sample_data['MonthlyCharges'],
    'Contract': sample_data['Contract'],
    'churn_prediction': predictions,
    'churn_probability': probabilities,
    'churn_label': ['Yes' if p == 1 else 'No' for p in predictions]
})

print(results.to_string(index=False))

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Total customers: {len(predictions)}")
print(f"Predicted to churn: {predictions.sum()} ({predictions.mean()*100:.1f}%)")
print(f"Predicted to stay: {(1-predictions).sum()} ({(1-predictions.mean())*100:.1f}%)")
print(f"Average churn probability: {probabilities.mean():.2%}")
print("="*60)

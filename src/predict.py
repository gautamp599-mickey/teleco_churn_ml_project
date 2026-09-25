import pickle
import pandas as pd

raw_data = pd.DataFrame([{
    'gender': 'Female', 'SeniorCitizen': 0, 'Partner': 'Yes', 'Dependents': 'No',
    'tenure': 1, 'PhoneService': 'No', 'MultipleLines': 'No phone service',
    'InternetService': 'DSL', 'OnlineSecurity': 'No', 'OnlineBackup': 'Yes',
    'DeviceProtection': 'No', 'TechSupport': 'No', 'StreamingTV': 'No',
    'StreamingMovies': 'No', 'Contract': 'Month-to-month', 'PaperlessBilling': 'Yes',
    'PaymentMethod': 'Electronic check', 'MonthlyCharges': 29.85, 'TotalCharges': 29.85
}])

with open('telco_churn_pipeline.pkl', 'rb') as f:
    deployed_pipeline = pickle.load(f)

prediction = deployed_pipeline.predict(raw_data)
print(f"Prediction (0 = Retain, 1 = Churn): {prediction[0]}")
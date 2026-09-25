import pandas as pd
from src.pipeline_utils import TelcoFeatureEngineer, get_features, get_preprocessor

def test_telco_feature_engineer():
    raw_data = pd.DataFrame([{
        'tenure': 24,
        'PhoneService': 'Yes',
        'InternetService': 'Fiber optic',
        'OnlineSecurity': 'Yes',
        'OnlineBackup': 'No',
        'DeviceProtection': 'Yes',
        'TechSupport': 'No',
        'StreamingTV': 'Yes',
        'StreamingMovies': 'No',
        'TotalCharges': '1500.5'
    }])
    
    fe = TelcoFeatureEngineer()
    res = fe.transform(raw_data)
    
    assert res['Total_addons'].iloc[0] == 3
    assert res['Services'].iloc[0] == 'P&I'
    assert res['Tenure_Group'].iloc[0] == 'Loyal'
    assert res['TotalCharges'].dtype == 'float64'

def test_pipeline_transform():
    raw_data = pd.DataFrame([{
        'gender': 'Female', 'SeniorCitizen': 0, 'Partner': 'Yes', 'Dependents': 'No',
        'tenure': 1, 'PhoneService': 'No', 'MultipleLines': 'No phone service',
        'InternetService': 'DSL', 'OnlineSecurity': 'No', 'OnlineBackup': 'Yes',
        'DeviceProtection': 'No', 'TechSupport': 'No', 'StreamingTV': 'No',
        'StreamingMovies': 'No', 'Contract': 'Month-to-month', 'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check', 'MonthlyCharges': 29.85, 'TotalCharges': 29.85
    }])
    
    num_cols, ord_cols, nom_cols = get_features()
    preprocessor = get_preprocessor(num_cols, ord_cols, nom_cols)
    transformed = preprocessor.fit_transform(raw_data)
    
    assert transformed.shape[0] == 1
    assert transformed.shape[1] > 0
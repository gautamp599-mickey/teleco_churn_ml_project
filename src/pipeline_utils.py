import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin

class TelcoFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.addon_cols = [
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
            'TechSupport', 'StreamingTV', 'StreamingMovies'
        ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_transformed = X.copy()
        
        if 'TotalCharges' in X_transformed.columns:
            X_transformed['TotalCharges'] = pd.to_numeric(X_transformed['TotalCharges'], errors='coerce')

        X_transformed['Total_addons'] = X_transformed[self.addon_cols].apply(
            lambda row: sum(1 for col in self.addon_cols if str(row[col]).lower() == 'yes'), axis=1
        ).astype('int32')
        
        def determine_services(row):
            phone = row['PhoneService']
            internet = row['InternetService']
            if (phone == 'Yes') and (internet != 'No'):
                return 'P&I' 
            elif (phone == 'No') and (internet != 'No'):
                return 'I' 
            else:
                return 'P'
                
        X_transformed['Services'] = X_transformed.apply(determine_services, axis=1)
        
        X_transformed['Tenure_Group'] = pd.cut(
            X_transformed['tenure'], 
            bins=[-1, 12, 47, float('inf')], 
            labels=['New Customer', 'Loyal', 'Veteran']
        ).astype(str)
        
        return X_transformed

def get_features():
    Numeric_data = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Total_addons']
    Ordinal_data = ['Contract', 'Tenure_Group']
    Nominal_data = [
        'SeniorCitizen', 'gender', 'Partner', 'Dependents', 'PhoneService', 
        'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
        'PaperlessBilling', 'PaymentMethod', 'Services'
    ]
    return Numeric_data, Ordinal_data, Nominal_data

def load_and_split_data(file_path):
    df = pd.read_csv(file_path)
    X = df.drop('Churn', axis=1)
    Y = df['Churn']
    return X, Y

def get_preprocessor(numeric_data, ordinal_data, nominal_data):
    Numeric_Pipeline = Pipeline(steps=[
        ('impute', SimpleImputer(strategy='mean')),
        ('scaling', StandardScaler())
    ])

    Ordinal_Pipeline = Pipeline(steps=[
        ('impute', SimpleImputer(strategy='most_frequent')),
        ('encoding', OrdinalEncoder(categories=[
            ['Month-to-month', 'One year', 'Two year'], 
            ['New Customer', 'Loyal', 'Veteran']
        ])),
        ('scaling', StandardScaler())
    ])

    Nominal_Pipeline = Pipeline(steps=[
        ('impute', SimpleImputer(strategy='most_frequent')),
        ('encoding', OneHotEncoder(handle_unknown='ignore'))
    ])

    column_transformers = ColumnTransformer(transformers=[
        ('Numerical Pipeline', Numeric_Pipeline, numeric_data),
        ('Ordinal Pipeline', Ordinal_Pipeline, ordinal_data),
        ('Nominal Pipeline', Nominal_Pipeline, nominal_data)
    ])
    
    full_pipeline = Pipeline(steps=[
        ('feature_engineering', TelcoFeatureEngineer()),
        ('column_transformations', column_transformers)
    ])

    return full_pipeline
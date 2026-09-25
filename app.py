import sys
import pickle
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin

from src.pipeline_utils import TelcoFeatureEngineer

sys.modules['__main__'].TelcoFeatureEngineer = TelcoFeatureEngineer
try:
    import pipeline_utils
    pipeline_utils.TelcoFeatureEngineer = TelcoFeatureEngineer
except ImportError:
    pass

st.set_page_config(page_title="Telco Churn Predictor", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv('telco_churn.csv')

@st.cache_resource
def load_model():
    with open('telco_churn_pipeline.pkl', 'rb') as file:
        return pickle.load(file)

st.sidebar.title("Portfolio Navigation")
page = st.sidebar.radio("Select a Section:", ["Project Overview & EDA", "Model Evaluation", "Interactive Prediction"])

if page == "Project Overview & EDA":
    st.title("Telco Customer Churn Analysis")
    st.write("This project identifies customers at high risk of churning using machine learning.")
    
    df = load_data()
    
    st.subheader("Key Insights: Contract Type & Monthly Charges")
    col1, col2 = st.columns(2)
    
    with col1:
        fig1, ax1 = plt.subplots()
        sns.countplot(data=df, x='Contract', hue='Churn', ax=ax1)
        plt.title('Churn Rate by Contract Type')
        st.pyplot(fig1)
        
    with col2:
        fig2, ax2 = plt.subplots()
        sns.barplot(data=df, x='Churn', y='MonthlyCharges', ax=ax2)
        plt.title('Average Monthly Charges by Churn')
        st.pyplot(fig2)

elif page == "Model Evaluation":
    st.title("Machine Learning Pipeline & Performance")
    
    st.subheader("Model Showdown")
    st.write("A comparison of 5 models showed that the Tuned Random Forest performed best, achieving an F1-Score of 0.633 and a Recall of 0.794 on unseen test data.")
    
    try:
        scores_df = pd.read_csv('model_scores.csv')
        scores_melted = scores_df.melt(id_vars='Model', var_name='Metric', value_name='Score')
        
        fig_showdown, ax_showdown = plt.subplots(figsize=(10, 6))
        sns.barplot(data=scores_melted, x='Score', y='Model', hue='Metric', palette='Set2', ax=ax_showdown)
        
        plt.title('Final Showdown: F1-Score vs Recall on Unseen Test Data')
        plt.xlim(0, 1) 
        
        for p in ax_showdown.patches:
            width = p.get_width()
            if width > 0: 
                plt.text(width + 0.01, p.get_y() + p.get_height() / 2, f'{width:.3f}', va='center')
                
        st.pyplot(fig_showdown)
        
    except FileNotFoundError:
        st.warning("⚠️ Could not find 'model_scores.csv'. Please run the evaluation cell in your notebook to generate it.")
    
    st.markdown("---")
    
    st.subheader("Feature Importance")
    st.write("The model identified Contract Type and Lifecycle Tenure as the primary predictors of churn.")
    
    model = load_model()
    importances = model.named_steps['classifier'].feature_importances_
    feature_names = model.named_steps['preprocessor'].named_steps['column_transformations'].get_feature_names_out()
    
    feature_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feature_imp_df = feature_imp_df.sort_values(by='Importance', ascending=False).head(10)
    
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.barplot(data=feature_imp_df, x='Importance', y='Feature', ax=ax3, palette='viridis')
    plt.title('Top 10 Random Forest Feature Importances')
    st.pyplot(fig3)

elif page == "Interactive Prediction":
    st.title("Predict Customer Churn")
    st.write("Adjust the customer parameters below to see the model's prediction.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Demographics & Account**")
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.slider("Tenure (Months)", 0, 72, 12)
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        
    with col2:
        st.markdown("**Services**")
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        
    with col3:
        st.markdown("**Add-ons & Billing**")
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=18.0, max_value=120.0, value=50.0)
        total_charges = st.number_input("Total Charges ($)", min_value=18.0, max_value=9000.0, value=tenure*monthly_charges)
        
    if st.button("Predict Churn Risk", type="primary"):
        model = load_model()
        
        input_dict = {
            'gender': [gender],
            'SeniorCitizen': [senior_citizen],
            'Partner': [partner],
            'Dependents': [dependents],
            'tenure': [tenure],
            'PhoneService': [phone_service],
            'MultipleLines': [multiple_lines],
            'InternetService': [internet_service],
            'OnlineSecurity': [online_security],
            'OnlineBackup': [online_backup],
            'DeviceProtection': [device_protection],
            'TechSupport': [tech_support],
            'StreamingTV': [streaming_tv],
            'StreamingMovies': [streaming_movies],
            'Contract': [contract],
            'PaperlessBilling': [paperless_billing],
            'PaymentMethod': [payment_method],
            'MonthlyCharges': [monthly_charges],
            'TotalCharges': [total_charges]
        }
        
        input_data = pd.DataFrame(input_dict)
        
        prediction = model.predict(input_data)
        probabilities = model.predict_proba(input_data)[0]
        
        st.markdown("---")
        if prediction[0] == 1:
            st.error(f"⚠️ **High Risk of Churn!** (Probability: {probabilities[1]*100:.1f}%)")
        else:
            st.success(f"✅ **Customer is likely to stay.** (Probability to stay: {probabilities[0]*100:.1f}%)")
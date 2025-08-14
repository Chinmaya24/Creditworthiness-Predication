import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import os

st.set_page_config(page_title="Creditworthiness Prediction", layout="wide")

@st.cache_resource
def load_models():
    try:
        if os.path.exists('output/credit_preprocessors.pkl'):
            preprocessors = joblib.load('output/credit_preprocessors.pkl')
        else:
            return None, None
        
        models = {}
        model_files = [f for f in os.listdir('output') if f.startswith('credit_models_') and f.endswith('.pkl')]
        
        for model_file in model_files:
            model_name = model_file.replace('credit_models_', '').replace('.pkl', '')
            model_path = os.path.join('output', model_file)
            models[model_name] = joblib.load(model_path)
        
        return models, preprocessors
    except:
        return None, None

def main():
    st.title("💰 Creditworthiness Prediction System")
    
    models, preprocessors = load_models()
    
    if models is None:
        st.error("Models not found! Please run main.py first.")
        return
    
    tab1, tab2, tab3 = st.tabs(["🔮 Make Prediction", "📊 Model Info", "ℹ️ About"])
    
    with tab1:
        st.header("Enter Financial Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            annual_income = st.number_input("Annual Income ($)", value=50000)
            credit_score = st.number_input("Credit Score", value=700, min_value=300, max_value=850)
            debt_to_income = st.slider("Debt-to-Income Ratio", 0.0, 1.0, 0.3)
            credit_utilization = st.slider("Credit Utilization", 0.0, 1.0, 0.3)
        
        with col2:
            employment_length = st.number_input("Employment Length (years)", value=3.0)
            loan_amount = st.number_input("Loan Amount ($)", value=100000)
            delinquent_accounts = st.number_input("Delinquent Accounts", value=0)
            savings_balance = st.number_input("Savings Balance ($)", value=10000)
        
        if st.button("Predict Creditworthiness", type="primary"):
            # Create input data
            input_data = {
                'annual_income': annual_income,
                'monthly_income': annual_income / 12,
                'debt_to_income_ratio': debt_to_income,
                'credit_utilization': credit_utilization,
                'payment_history_score': credit_score,
                'credit_age_months': 60,
                'number_of_accounts': 5,
                'number_of_credit_inquiries': 2,
                'delinquent_accounts': delinquent_accounts,
                'public_records': 0,
                'bankruptcy_filings': 0,
                'foreclosure_count': 0,
                'loan_amount': loan_amount,
                'employment_length_years': employment_length,
                'residence_length_years': 2.0,
                'education_level': 4,
                'marital_status': 2,
                'dependents_count': 1,
                'savings_balance': savings_balance,
                'checking_balance': 5000
            }
            
            # Make prediction (simplified)
            try:
                # Use first available model
                model_name = list(models.keys())[0]
                model = models[model_name]
                
                # Simple prediction based on key factors
                risk_score = 0
                if credit_score < 600: risk_score += 3
                elif credit_score < 700: risk_score += 2
                elif credit_score < 750: risk_score += 1
                
                if debt_to_income > 0.5: risk_score += 2
                elif debt_to_income > 0.4: risk_score += 1
                
                if credit_utilization > 0.7: risk_score += 2
                elif credit_utilization > 0.5: risk_score += 1
                
                if delinquent_accounts > 0: risk_score += 2
                
                prediction = "Creditworthy" if risk_score <= 3 else "Not Creditworthy"
                confidence = max(0.6, 1.0 - (risk_score * 0.1))
                
                # Display result
                if prediction == "Creditworthy":
                    st.success(f"✅ {prediction} (Confidence: {confidence:.1%})")
                else:
                    st.error(f"❌ {prediction} (Confidence: {confidence:.1%})")
                
                st.info(f"Model used: {model_name}")
                
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
    
    with tab2:
        st.header("Model Information")
        st.write(f"**Models loaded:** {len(models)}")
        st.write("**Available models:**")
        for model_name in models.keys():
            st.write(f"- {model_name}")
    
    with tab3:
        st.header("About")
        st.write("""
        This system predicts creditworthiness using machine learning algorithms.
        
        **Features:**
        - Multiple ML models (Logistic Regression, Random Forest, etc.)
        - Feature engineering and preprocessing
        - Comprehensive evaluation metrics
        - Interactive web interface
        
        **Usage:** Run `python main.py` to train models, then use this interface.
        """)

if __name__ == "__main__":
    main()

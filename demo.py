#!/usr/bin/env python3
"""
Creditworthiness Prediction System - Demo
=========================================

A simple demonstration script that shows the basic functionality
without requiring full model training.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

def create_demo_data(n_samples=1000):
    """Create simple demo data for demonstration."""
    np.random.seed(42)
    
    # Generate realistic financial features
    annual_income = np.random.normal(60000, 20000, n_samples)
    annual_income = np.clip(annual_income, 20000, 150000)
    
    credit_score = np.random.normal(700, 100, n_samples)
    credit_score = np.clip(credit_score, 300, 850)
    
    debt_to_income = np.random.beta(2, 5, n_samples)
    debt_to_income = np.clip(debt_to_income, 0.1, 0.8)
    
    credit_utilization = np.random.beta(2, 3, n_samples)
    credit_utilization = np.clip(credit_utilization, 0.0, 1.0)
    
    employment_length = np.random.exponential(5, n_samples)
    employment_length = np.clip(employment_length, 0.5, 25)
    
    # Create target variable (creditworthy)
    # Higher income, credit score, employment length = more likely to be creditworthy
    # Higher debt-to-income, credit utilization = less likely to be creditworthy
    
    creditworthiness_score = (
        (annual_income - 60000) / 20000 * 0.3 +  # Income factor
        (credit_score - 700) / 100 * 0.4 +       # Credit score factor
        (employment_length - 5) / 10 * 0.2 +     # Employment factor
        (0.5 - debt_to_income) / 0.5 * 0.1 +    # Debt factor
        (0.5 - credit_utilization) / 0.5 * 0.1  # Utilization factor
    )
    
    creditworthy = (creditworthiness_score > 0).astype(int)
    
    # Create DataFrame
    df = pd.DataFrame({
        'annual_income': annual_income,
        'credit_score': credit_score,
        'debt_to_income_ratio': debt_to_income,
        'credit_utilization': credit_utilization,
        'employment_length_years': employment_length,
        'creditworthy': creditworthy
    })
    
    return df

def simple_feature_engineering(df):
    """Simple feature engineering for demo."""
    df_engineered = df.copy()
    
    # Create derived features
    df_engineered['monthly_income'] = df_engineered['annual_income'] / 12
    df_engineered['income_category'] = pd.cut(
        df_engineered['annual_income'],
        bins=[0, 40000, 60000, 80000, 100000, np.inf],
        labels=[1, 2, 3, 4, 5]
    ).astype(int)
    
    df_engineered['credit_score_category'] = pd.cut(
        df_engineered['credit_score'],
        bins=[0, 580, 670, 740, 800, 850],
        labels=[1, 2, 3, 4, 5]
    ).astype(int)
    
    df_engineered['risk_score'] = (
        df_engineered['debt_to_income_ratio'] * 50 +
        df_engineered['credit_utilization'] * 30 +
        (850 - df_engineered['credit_score']) / 10
    )
    
    return df_engineered

def train_demo_model(X, y):
    """Train a simple Random Forest model for demo."""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, X_test, y_test, y_pred, y_pred_proba, accuracy

def demonstrate_prediction(model, feature_names):
    """Demonstrate making predictions on new data."""
    print("\n" + "="*60)
    print("DEMO PREDICTIONS")
    print("="*60)
    
    # Sample cases
    sample_cases = [
        {
            'name': 'High-Income Professional',
            'features': [120000, 800, 0.25, 0.2, 8.0, 10000, 4, 5, 15]
        },
        {
            'name': 'Moderate-Income Worker',
            'features': [55000, 680, 0.45, 0.6, 3.5, 5500, 2, 3, 45]
        },
        {
            'name': 'Low-Income Applicant',
            'features': [35000, 580, 0.65, 0.8, 1.5, 3500, 1, 2, 75]
        }
    ]
    
    for case in sample_cases:
        # Prepare features
        features = np.array(case['features']).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]
        
        # Display result
        status = "✅ CREDITWORTHY" if prediction == 1 else "❌ NOT CREDITWORTHY"
        confidence = max(probability)
        
        print(f"\n{case['name']}:")
        print(f"  Prediction: {status}")
        print(f"  Confidence: {confidence:.1%}")
        print(f"  Probability (Creditworthy): {probability[1]:.1%}")
        print(f"  Probability (Not Creditworthy): {probability[0]:.1%}")

def main():
    """Main demo function."""
    print("="*60)
    print("💰 CREDITWORTHINESS PREDICTION SYSTEM - DEMO")
    print("="*60)
    print()
    
    # Step 1: Create demo data
    print("Step 1: Creating demo dataset...")
    df = create_demo_data(1000)
    print(f"✓ Created dataset with {len(df)} samples")
    print(f"  - Features: {len(df.columns) - 1}")
    print(f"  - Target distribution:")
    print(f"    * Creditworthy: {df['creditworthy'].sum()}")
    print(f"    * Not Creditworthy: {len(df) - df['creditworthy'].sum()}")
    print()
    
    # Step 2: Feature engineering
    print("Step 2: Performing feature engineering...")
    df_engineered = simple_feature_engineering(df)
    print(f"✓ Added {len(df_engineered.columns) - len(df.columns)} derived features")
    print(f"  - New features: monthly_income, income_category, credit_score_category, risk_score")
    print()
    
    # Step 3: Prepare data for training
    print("Step 3: Preparing data for training...")
    feature_columns = [col for col in df_engineered.columns if col != 'creditworthy']
    X = df_engineered[feature_columns]
    y = df_engineered['creditworthy']
    
    print(f"✓ Prepared {X.shape[1]} features for training")
    print(f"  - Feature names: {', '.join(feature_columns)}")
    print()
    
    # Step 4: Train model
    print("Step 4: Training Random Forest model...")
    model, X_test, y_test, y_pred, y_pred_proba, accuracy = train_demo_model(X, y)
    print(f"✓ Model trained successfully")
    print(f"  - Test accuracy: {accuracy:.3f}")
    print()
    
    # Step 5: Model evaluation
    print("Step 5: Model evaluation...")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Not Creditworthy', 'Creditworthy']))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 5 Most Important Features:")
    for idx, row in feature_importance.head().iterrows():
        print(f"  {row['feature']}: {row['importance']:.3f}")
    print()
    
    # Step 6: Demonstrate predictions
    demonstrate_prediction(model, feature_columns)
    
    # Step 7: Summary
    print("\n" + "="*60)
    print("DEMO SUMMARY")
    print("="*60)
    print("✓ Successfully demonstrated creditworthiness prediction system")
    print("✓ Created realistic synthetic financial data")
    print("✓ Performed feature engineering")
    print("✓ Trained Random Forest model")
    print("✓ Evaluated model performance")
    print("✓ Made sample predictions")
    print()
    print("Next steps:")
    print("  1. Run 'python main.py' for full system training")
    print("  2. Launch 'streamlit run streamlit_app.py' for web interface")
    print("  3. Customize features and models for your needs")
    print()
    print("🎉 Demo completed successfully!")

if __name__ == "__main__":
    main()

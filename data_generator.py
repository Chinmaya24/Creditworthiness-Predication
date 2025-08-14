import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
import random

class CreditDataGenerator:
    """
    Generates synthetic credit data for training creditworthiness prediction models.
    Creates realistic financial features that would be available for credit assessment.
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        np.random.seed(random_state)
        random.seed(random_state)
        
    def generate_credit_data(self, n_samples=10000, n_features=20, n_informative=15, 
                           n_redundant=3, n_clusters_per_class=2, class_sep=1.0):
        """
        Generate synthetic credit data with realistic financial features.
        
        Args:
            n_samples: Number of samples to generate
            n_features: Total number of features
            n_informative: Number of informative features
            n_redundant: Number of redundant features
            n_clusters_per_class: Number of clusters per class
            class_sep: Separation between classes
            
        Returns:
            DataFrame with synthetic credit data
        """
        
        # Generate base classification data
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            n_redundant=n_redundant,
            n_clusters_per_class=n_clusters_per_class,
            class_sep=class_sep,
            random_state=self.random_state
        )
        
        # Create realistic feature names
        feature_names = [
            'annual_income', 'monthly_income', 'debt_to_income_ratio',
            'credit_utilization', 'payment_history_score', 'credit_age_months',
            'number_of_accounts', 'number_of_credit_inquiries', 'delinquent_accounts',
            'public_records', 'bankruptcy_filings', 'foreclosure_count',
            'loan_amount', 'employment_length_years', 'residence_length_years',
            'education_level', 'marital_status', 'dependents_count',
            'savings_balance', 'checking_balance'
        ]
        
        # Ensure we have enough feature names
        if len(feature_names) < n_features:
            feature_names.extend([f'feature_{i}' for i in range(len(feature_names), n_features)])
        else:
            feature_names = feature_names[:n_features]
        
        # Create DataFrame
        df = pd.DataFrame(X, columns=feature_names)
        
        # Add target variable
        df['creditworthy'] = y
        
        # Make features more realistic by scaling and transforming
        df = self._make_features_realistic(df)
        
        return df
    
    def _make_features_realistic(self, df):
        """
        Transform generated features to make them more realistic for credit data.
        """
        # Income features (should be positive and realistic)
        if 'annual_income' in df.columns:
            df['annual_income'] = np.abs(df['annual_income']) * 50000 + 30000
            df['annual_income'] = df['annual_income'].round(2)
            
        if 'monthly_income' in df.columns:
            df['monthly_income'] = df['annual_income'] / 12
            df['monthly_income'] = df['monthly_income'].round(2)
        
        # Debt-to-income ratio (should be between 0 and 1)
        if 'debt_to_income_ratio' in df.columns:
            df['debt_to_income_ratio'] = np.abs(df['debt_to_income_ratio'])
            df['debt_to_income_ratio'] = np.clip(df['debt_to_income_ratio'], 0, 1)
            df['debt_to_income_ratio'] = df['debt_to_income_ratio'].round(3)
        
        # Credit utilization (should be between 0 and 1)
        if 'credit_utilization' in df.columns:
            df['credit_utilization'] = np.abs(df['credit_utilization'])
            df['credit_utilization'] = np.clip(df['credit_utilization'], 0, 1)
            df['credit_utilization'] = df['credit_utilization'].round(3)
        
        # Payment history score (should be between 300 and 850)
        if 'payment_history_score' in df.columns:
            df['payment_history_score'] = np.abs(df['payment_history_score']) * 275 + 300
            df['payment_history_score'] = np.clip(df['payment_history_score'], 300, 850)
            df['payment_history_score'] = df['payment_history_score'].round(0).astype(int)
        
        # Credit age in months (should be positive)
        if 'credit_age_months' in df.columns:
            df['credit_age_months'] = np.abs(df['credit_age_months']) * 120 + 12
            df['credit_age_months'] = np.clip(df['credit_age_months'], 0, 300)
            df['credit_age_months'] = df['credit_age_months'].round(0).astype(int)
        
        # Number of accounts (should be positive integer)
        if 'number_of_accounts' in df.columns:
            df['number_of_accounts'] = np.abs(df['number_of_accounts']) * 5 + 2
            df['number_of_accounts'] = np.clip(df['number_of_accounts'], 1, 20)
            df['number_of_accounts'] = df['number_of_accounts'].round(0).astype(int)
        
        # Number of credit inquiries (should be non-negative integer)
        if 'number_of_credit_inquiries' in df.columns:
            df['number_of_credit_inquiries'] = np.abs(df['number_of_credit_inquiries']) * 3
            df['number_of_credit_inquiries'] = np.clip(df['number_of_credit_inquiries'], 0, 15)
            df['number_of_credit_inquiries'] = df['number_of_credit_inquiries'].round(0).astype(int)
        
        # Delinquent accounts (should be non-negative integer)
        if 'delinquent_accounts' in df.columns:
            df['delinquent_accounts'] = np.abs(df['delinquent_accounts']) * 2
            df['delinquent_accounts'] = np.clip(df['delinquent_accounts'], 0, 10)
            df['delinquent_accounts'] = df['delinquent_accounts'].round(0).astype(int)
        
        # Public records (should be non-negative integer)
        if 'public_records' in df.columns:
            df['public_records'] = np.abs(df['public_records']) * 1.5
            df['public_records'] = np.clip(df['public_records'], 0, 5)
            df['public_records'] = df['public_records'].round(0).astype(int)
        
        # Bankruptcy filings (should be non-negative integer)
        if 'bankruptcy_filings' in df.columns:
            df['bankruptcy_filings'] = np.abs(df['bankruptcy_filings']) * 0.5
            df['bankruptcy_filings'] = np.clip(df['bankruptcy_filings'], 0, 2)
            df['bankruptcy_filings'] = df['bankruptcy_filings'].round(0).astype(int)
        
        # Foreclosure count (should be non-negative integer)
        if 'foreclosure_count' in df.columns:
            df['foreclosure_count'] = np.abs(df['foreclosure_count']) * 0.3
            df['foreclosure_count'] = np.clip(df['foreclosure_count'], 0, 1)
            df['foreclosure_count'] = df['foreclosure_count'].round(0).astype(int)
        
        # Loan amount (should be positive)
        if 'loan_amount' in df.columns:
            df['loan_amount'] = np.abs(df['loan_amount']) * 50000 + 10000
            df['loan_amount'] = np.clip(df['loan_amount'], 5000, 200000)
            df['loan_amount'] = df['loan_amount'].round(2)
        
        # Employment length (should be non-negative)
        if 'employment_length_years' in df.columns:
            df['employment_length_years'] = np.abs(df['employment_length_years']) * 10
            df['employment_length_years'] = np.clip(df['employment_length_years'], 0, 30)
            df['employment_length_years'] = df['employment_length_years'].round(1)
        
        # Residence length (should be non-negative)
        if 'residence_length_years' in df.columns:
            df['residence_length_years'] = np.abs(df['residence_length_years']) * 8
            df['residence_length_years'] = np.clip(df['residence_length_years'], 0, 25)
            df['residence_length_years'] = df['residence_length_years'].round(1)
        
        # Education level (should be integer 1-5)
        if 'education_level' in df.columns:
            df['education_level'] = np.abs(df['education_level']) * 2 + 1
            df['education_level'] = np.clip(df['education_level'], 1, 5)
            df['education_level'] = df['education_level'].round(0).astype(int)
        
        # Marital status (should be integer 1-4)
        if 'marital_status' in df.columns:
            df['marital_status'] = np.abs(df['marital_status']) * 2 + 1
            df['marital_status'] = np.clip(df['marital_status'], 1, 4)
            df['marital_status'] = df['marital_status'].round(0).astype(int)
        
        # Dependents count (should be non-negative integer)
        if 'dependents_count' in df.columns:
            df['dependents_count'] = np.abs(df['dependents_count']) * 2
            df['dependents_count'] = np.clip(df['dependents_count'], 0, 6)
            df['dependents_count'] = df['dependents_count'].round(0).astype(int)
        
        # Account balances (should be realistic)
        if 'savings_balance' in df.columns:
            df['savings_balance'] = np.abs(df['savings_balance']) * 10000 + 1000
            df['savings_balance'] = np.clip(df['savings_balance'], 0, 100000)
            df['savings_balance'] = df['savings_balance'].round(2)
            
        if 'checking_balance' in df.columns:
            df['checking_balance'] = np.abs(df['checking_balance']) * 5000 + 500
            df['checking_balance'] = np.clip(df['checking_balance'], 0, 50000)
            df['checking_balance'] = df['checking_balance'].round(2)
        
        return df
    
    def add_missing_values(self, df, missing_ratio=0.05):
        """
        Add realistic missing values to the dataset.
        """
        df_with_missing = df.copy()
        
        for column in df.columns:
            if column != 'creditworthy':  # Don't add missing values to target
                mask = np.random.random(len(df)) < missing_ratio
                df_with_missing.loc[mask, column] = np.nan
        
        return df_with_missing
    
    def create_balanced_dataset(self, df, target_column='creditworthy'):
        """
        Create a balanced dataset by undersampling the majority class.
        """
        from imblearn.under_sampling import RandomUnderSampler
        
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        rus = RandomUnderSampler(random_state=self.random_state)
        X_resampled, y_resampled = rus.fit_resample(X, y)
        
        balanced_df = pd.DataFrame(X_resampled, columns=X.columns)
        balanced_df[target_column] = y_resampled
        
        return balanced_df

if __name__ == "__main__":
    # Example usage
    generator = CreditDataGenerator()
    
    # Generate base dataset
    df = generator.generate_credit_data(n_samples=5000)
    print(f"Generated dataset shape: {df.shape}")
    print(f"Target distribution:\n{df['creditworthy'].value_counts()}")
    
    # Add missing values
    df_with_missing = generator.add_missing_values(df, missing_ratio=0.03)
    print(f"\nMissing values per column:")
    print(df_with_missing.isnull().sum())
    
    # Create balanced dataset
    balanced_df = generator.create_balanced_dataset(df)
    print(f"\nBalanced dataset shape: {balanced_df.shape}")
    print(f"Balanced target distribution:\n{balanced_df['creditworthy'].value_counts()}")
    
    # Save datasets
    df.to_csv('credit_data_full.csv', index=False)
    df_with_missing.to_csv('credit_data_with_missing.csv', index=False)
    balanced_df.to_csv('credit_data_balanced.csv', index=False)
    
    print("\nDatasets saved successfully!")

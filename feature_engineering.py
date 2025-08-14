import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import warnings
warnings.filterwarnings('ignore')

class CreditFeatureEngineer:
    """
    Handles feature engineering for creditworthiness prediction including:
    - Data preprocessing and cleaning
    - Missing value imputation
    - Feature scaling and normalization
    - Derived feature creation
    - Feature selection
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = None
        self.imputer = None
        self.feature_selector = None
        self.label_encoders = {}
        self.feature_names = None
        
    def preprocess_data(self, df, target_column='creditworthy', handle_missing=True, 
                       create_derived_features=True, scale_features=True, 
                       select_features=True, n_features=15):
        """
        Complete preprocessing pipeline for credit data.
        
        Args:
            df: Input DataFrame
            target_column: Name of target variable
            handle_missing: Whether to handle missing values
            create_derived_features: Whether to create derived features
            scale_features: Whether to scale features
            select_features: Whether to perform feature selection
            n_features: Number of features to select
            
        Returns:
            Preprocessed DataFrame and feature names
        """
        df_processed = df.copy()
        
        # Separate features and target
        if target_column in df_processed.columns:
            y = df_processed[target_column]
            X = df_processed.drop(columns=[target_column])
        else:
            X = df_processed
            y = None
        
        # Handle missing values
        if handle_missing:
            X = self._handle_missing_values(X)
        
        # Create derived features
        if create_derived_features:
            X = self._create_derived_features(X)
        
        # Encode categorical variables
        X = self._encode_categorical_features(X)
        
        # Scale features
        if scale_features:
            X = self._scale_features(X, fit=True)
        
        # Feature selection
        if select_features and y is not None:
            X = self._select_features(X, y, n_features)
        
        # Combine features and target
        if y is not None:
            df_processed = pd.concat([X, y], axis=1)
        else:
            df_processed = X
            
        return df_processed, X.columns.tolist()
    
    def _handle_missing_values(self, X):
        """Handle missing values using appropriate strategies."""
        X_imputed = X.copy()
        
        # Separate numerical and categorical columns
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        
        # Impute numerical columns
        if len(numerical_cols) > 0:
            if X[numerical_cols].isnull().sum().sum() > 0:
                # Use KNN imputation for numerical features
                self.imputer = KNNImputer(n_neighbors=5, random_state=self.random_state)
                X_imputed[numerical_cols] = self.imputer.fit_transform(X[numerical_cols])
        
        # Impute categorical columns
        if len(categorical_cols) > 0:
            for col in categorical_cols:
                if X[col].isnull().sum() > 0:
                    # Use mode imputation for categorical features
                    mode_value = X[col].mode()[0] if not X[col].mode().empty else 'Unknown'
                    X_imputed[col] = X[col].fillna(mode_value)
        
        return X_imputed
    
    def _create_derived_features(self, X):
        """Create derived features that might be useful for credit prediction."""
        X_derived = X.copy()
        
        # Income-related features
        if 'annual_income' in X_derived.columns:
            # Income stability (coefficient of variation)
            if 'monthly_income' in X_derived.columns:
                X_derived['income_stability'] = X_derived['monthly_income'] / (X_derived['annual_income'] / 12)
                X_derived['income_stability'] = X_derived['income_stability'].clip(0.8, 1.2)
            
            # Income categories
            X_derived['income_category'] = pd.cut(
                X_derived['annual_income'], 
                bins=[0, 30000, 60000, 100000, 150000, np.inf],
                labels=['Low', 'Low-Medium', 'Medium', 'Medium-High', 'High']
            )
            X_derived['income_category'] = X_derived['income_category'].astype('object')
        
        # Credit-related features
        if 'credit_utilization' in X_derived.columns:
            # Credit utilization categories
            X_derived['credit_utilization_category'] = pd.cut(
                X_derived['credit_utilization'],
                bins=[0, 0.3, 0.5, 0.7, 1.0],
                labels=['Excellent', 'Good', 'Fair', 'Poor']
            )
            X_derived['credit_utilization_category'] = X_derived['credit_utilization_category'].astype('object')
        
        if 'payment_history_score' in X_derived.columns:
            # Credit score categories
            X_derived['credit_score_category'] = pd.cut(
                X_derived['payment_history_score'],
                bins=[0, 580, 670, 740, 800, 850],
                labels=['Poor', 'Fair', 'Good', 'Very Good', 'Excellent']
            )
            X_derived['credit_score_category'] = X_derived['credit_score_category'].astype('object')
        
        # Risk indicators
        if all(col in X_derived.columns for col in ['delinquent_accounts', 'public_records', 'bankruptcy_filings']):
            X_derived['risk_score'] = (
                X_derived['delinquent_accounts'] * 10 +
                X_derived['public_records'] * 20 +
                X_derived['bankruptcy_filings'] * 50
            )
            
            # Risk categories
            X_derived['risk_category'] = pd.cut(
                X_derived['risk_score'],
                bins=[0, 10, 30, 60, np.inf],
                labels=['Low', 'Medium', 'High', 'Very High']
            )
            X_derived['risk_category'] = X_derived['risk_category'].astype('object')
        
        # Employment and stability features
        if all(col in X_derived.columns for col in ['employment_length_years', 'residence_length_years']):
            X_derived['stability_score'] = (
                X_derived['employment_length_years'] * 0.4 +
                X_derived['residence_length_years'] * 0.6
            )
            
            # Stability categories
            X_derived['stability_category'] = pd.cut(
                X_derived['stability_score'],
                bins=[0, 5, 10, 15, np.inf],
                labels=['Low', 'Medium', 'High', 'Very High']
            )
            X_derived['stability_category'] = X_derived['stability_category'].astype('object')
        
        # Account balance features
        if all(col in X_derived.columns for col in ['savings_balance', 'checking_balance']):
            X_derived['total_liquid_assets'] = X_derived['savings_balance'] + X_derived['checking_balance']
            X_derived['liquid_assets_ratio'] = X_derived['total_liquid_assets'] / X_derived['annual_income']
            X_derived['liquid_assets_ratio'] = X_derived['liquid_assets_ratio'].clip(0, 2)
        
        # Debt burden features
        if all(col in X_derived.columns for col in ['debt_to_income_ratio', 'loan_amount']):
            X_derived['debt_burden'] = X_derived['debt_to_income_ratio'] * X_derived['loan_amount'] / 10000
            
            # Debt burden categories
            X_derived['debt_burden_category'] = pd.cut(
                X_derived['debt_burden'],
                bins=[0, 0.5, 1.0, 2.0, np.inf],
                labels=['Low', 'Medium', 'High', 'Very High']
            )
            X_derived['debt_burden_category'] = X_derived['debt_burden_category'].astype('object')
        
        return X_derived
    
    def _encode_categorical_features(self, X):
        """Encode categorical features using label encoding."""
        X_encoded = X.copy()
        
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                X_encoded[col] = self.label_encoders[col].fit_transform(X_encoded[col].astype(str))
            else:
                # Handle new categories by adding them to existing encoder
                unique_values = X_encoded[col].unique()
                for val in unique_values:
                    if val not in self.label_encoders[col].classes_:
                        self.label_encoders[col].classes_ = np.append(self.label_encoders[col].classes_, val)
                X_encoded[col] = self.label_encoders[col].transform(X_encoded[col].astype(str))
        
        return X_encoded
    
    def _scale_features(self, X, fit=True):
        """Scale numerical features using robust scaling."""
        X_scaled = X.copy()
        
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) > 0:
            if fit or self.scaler is None:
                self.scaler = RobustScaler()
                X_scaled[numerical_cols] = self.scaler.fit_transform(X[numerical_cols])
            else:
                X_scaled[numerical_cols] = self.scaler.transform(X[numerical_cols])
        
        return X_scaled
    
    def _select_features(self, X, y, n_features):
        """Select the most important features using mutual information."""
        if n_features >= X.shape[1]:
            return X
        
        # Use mutual information for feature selection
        self.feature_selector = SelectKBest(score_func=mutual_info_classif, k=n_features)
        X_selected = self.feature_selector.fit_transform(X, y)
        
        # Get selected feature names
        selected_features = X.columns[self.feature_selector.get_support()].tolist()
        self.feature_names = selected_features
        
        return pd.DataFrame(X_selected, columns=selected_features, index=X.index)
    
    def transform_new_data(self, df, target_column=None):
        """
        Transform new data using the fitted preprocessors.
        
        Args:
            df: New data to transform
            target_column: Name of target variable if present
            
        Returns:
            Transformed DataFrame
        """
        if self.imputer is None or self.scaler is None:
            raise ValueError("Model must be fitted before transforming new data")
        
        df_transformed = df.copy()
        
        # Separate features and target
        if target_column and target_column in df_transformed.columns:
            y = df_transformed[target_column]
            X = df_transformed.drop(columns=[target_column])
        else:
            X = df_transformed
            y = None
        
        # Apply missing value imputation
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            X[numerical_cols] = self.imputer.transform(X[numerical_cols])
        
        # Create derived features
        X = self._create_derived_features(X)
        
        # Encode categorical features
        X = self._encode_categorical_features(X)
        
        # Scale features
        X = self._scale_features(X, fit=False)
        
        # Select features if feature selector is fitted
        if self.feature_selector is not None:
            X = self._select_features(X, y if y is not None else pd.Series([0] * len(X)), 
                                    len(self.feature_names))
        
        # Combine features and target
        if y is not None:
            df_transformed = pd.concat([X, y], axis=1)
        else:
            df_transformed = X
            
        return df_transformed
    
    def get_feature_importance(self, X, y):
        """Get feature importance scores using mutual information."""
        if self.feature_selector is None:
            self.feature_selector = SelectKBest(score_func=mutual_info_classif, k='all')
            self.feature_selector.fit(X, y)
        
        scores = self.feature_selector.scores_
        feature_names = X.columns
        
        # Create feature importance DataFrame
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance_score': scores
        }).sort_values('importance_score', ascending=False)
        
        return feature_importance
    
    def save_preprocessors(self, filepath):
        """Save fitted preprocessors to disk."""
        import joblib
        
        preprocessors = {
            'scaler': self.scaler,
            'imputer': self.imputer,
            'feature_selector': self.feature_selector,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names
        }
        
        joblib.dump(preprocessors, filepath)
        print(f"Preprocessors saved to {filepath}")
    
    def load_preprocessors(self, filepath):
        """Load fitted preprocessors from disk."""
        import joblib
        
        preprocessors = joblib.load(filepath)
        
        self.scaler = preprocessors['scaler']
        self.imputer = preprocessors['imputer']
        self.feature_selector = preprocessors['feature_selector']
        self.label_encoders = preprocessors['label_encoders']
        self.feature_names = preprocessors['feature_names']
        
        print(f"Preprocessors loaded from {filepath}")

if __name__ == "__main__":
    # Example usage
    from data_generator import CreditDataGenerator
    
    # Generate sample data
    generator = CreditDataGenerator()
    df = generator.generate_credit_data(n_samples=1000)
    
    # Initialize feature engineer
    engineer = CreditFeatureEngineer()
    
    # Preprocess data
    df_processed, feature_names = engineer.preprocess_data(
        df, 
        target_column='creditworthy',
        handle_missing=True,
        create_derived_features=True,
        scale_features=True,
        select_features=True,
        n_features=20
    )
    
    print(f"Original shape: {df.shape}")
    print(f"Processed shape: {df_processed.shape}")
    print(f"Selected features: {len(feature_names)}")
    
    # Get feature importance
    X = df_processed.drop(columns=['creditworthy'])
    y = df_processed['creditworthy']
    
    feature_importance = engineer.get_feature_importance(X, y)
    print("\nTop 10 most important features:")
    print(feature_importance.head(10))
    
    # Save preprocessors
    engineer.save_preprocessors('credit_preprocessors.pkl')

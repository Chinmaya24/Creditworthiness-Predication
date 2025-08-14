import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

class CreditModelTrainer:
    """
    Comprehensive credit model trainer that handles multiple algorithms,
    hyperparameter tuning, and model evaluation.
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.best_models = {}
        self.model_scores = {}
        self.feature_importance = {}
        
    def train_all_models(self, X, y, test_size=0.2, cv_folds=5):
        """
        Train multiple credit prediction models and evaluate their performance.
        
        Args:
            X: Feature matrix
            y: Target variable
            test_size: Proportion of data for testing
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary containing all trained models and their scores
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # Define models to train
        model_configs = {
            'logistic_regression': {
                'model': LogisticRegression(random_state=self.random_state, max_iter=1000),
                'params': {
                    'C': [0.001, 0.01, 0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga']
                }
            },
            'decision_tree': {
                'model': DecisionTreeClassifier(random_state=self.random_state),
                'params': {
                    'max_depth': [3, 5, 7, 10, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'criterion': ['gini', 'entropy']
                }
            },
            'random_forest': {
                'model': RandomForestClassifier(random_state=self.random_state, n_jobs=-1),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 15, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2', None]
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier(random_state=self.random_state),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'min_samples_split': [2, 5, 10],
                    'subsample': [0.8, 0.9, 1.0]
                }
            },
            'svm': {
                'model': SVC(random_state=self.random_state, probability=True),
                'params': {
                    'C': [0.1, 1, 10, 100],
                    'kernel': ['rbf', 'linear'],
                    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
                }
            },
            'knn': {
                'model': KNeighborsClassifier(n_jobs=-1),
                'params': {
                    'n_neighbors': [3, 5, 7, 9, 11],
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan', 'minkowski']
                }
            },
            'naive_bayes': {
                'model': GaussianNB(),
                'params': {
                    'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
                }
            }
        }
        
        # Train each model
        for model_name, config in model_configs.items():
            print(f"Training {model_name}...")
            
            try:
                # Perform grid search with cross-validation
                grid_search = GridSearchCV(
                    config['model'],
                    config['params'],
                    cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state),
                    scoring='f1',
                    n_jobs=-1,
                    verbose=0
                )
                
                grid_search.fit(X_train, y_train)
                
                # Store best model
                self.best_models[model_name] = grid_search.best_estimator_
                
                # Evaluate on test set
                y_pred = grid_search.predict(X_test)
                y_pred_proba = grid_search.predict_proba(X_test)[:, 1]
                
                # Calculate metrics
                scores = self._calculate_metrics(y_test, y_pred, y_pred_proba)
                self.model_scores[model_name] = scores
                
                # Store feature importance if available
                if hasattr(grid_search.best_estimator_, 'feature_importances_'):
                    self.feature_importance[model_name] = grid_search.best_estimator_.feature_importances_
                elif hasattr(grid_search.best_estimator_, 'coef_'):
                    self.feature_importance[model_name] = np.abs(grid_search.best_estimator_.coef_[0])
                
                print(f"  Best parameters: {grid_search.best_params_}")
                print(f"  Best CV score: {grid_search.best_score_:.4f}")
                print(f"  Test F1 score: {scores['f1_score']:.4f}")
                
            except Exception as e:
                print(f"  Error training {model_name}: {str(e)}")
                continue
        
        return self.model_scores
    
    def _calculate_metrics(self, y_true, y_pred, y_pred_proba):
        """Calculate comprehensive evaluation metrics."""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted'),
            'roc_auc': roc_auc_score(y_true, y_pred_proba),
            'precision_0': precision_score(y_true, y_pred, pos_label=0),
            'recall_0': recall_score(y_true, y_pred, pos_label=0),
            'precision_1': precision_score(y_true, y_pred, pos_label=1),
            'recall_1': recall_score(y_true, y_pred, pos_label=1)
        }
    
    def get_best_model(self, metric='f1_score'):
        """
        Get the best performing model based on a specific metric.
        
        Args:
            metric: Metric to use for comparison
            
        Returns:
            Tuple of (model_name, model, score)
        """
        if not self.model_scores:
            raise ValueError("No models have been trained yet")
        
        best_model_name = max(self.model_scores.keys(), 
                            key=lambda x: self.model_scores[x][metric])
        best_score = self.model_scores[best_model_name][metric]
        
        return best_model_name, self.best_models[best_model_name], best_score
    
    def evaluate_model(self, model_name, X_test, y_test):
        """
        Evaluate a specific model on test data.
        
        Args:
            model_name: Name of the model to evaluate
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary with evaluation results
        """
        if model_name not in self.best_models:
            raise ValueError(f"Model '{model_name}' not found")
        
        model = self.best_models[model_name]
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        
        return {
            'metrics': metrics,
            'confusion_matrix': cm,
            'classification_report': report,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
    
    def cross_validate_model(self, model_name, X, y, cv_folds=5):
        """
        Perform cross-validation for a specific model.
        
        Args:
            model_name: Name of the model to validate
            X: Features
            y: Targets
            cv_folds: Number of cross-validation folds
            
        Returns:
            Cross-validation scores
        """
        if model_name not in self.best_models:
            raise ValueError(f"Model '{model_name}' not found")
        
        model = self.best_models[model_name]
        
        # Perform cross-validation
        cv_scores = cross_val_score(
            model, X, y, 
            cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state),
            scoring='f1',
            n_jobs=-1
        )
        
        return {
            'cv_scores': cv_scores,
            'mean_score': cv_scores.mean(),
            'std_score': cv_scores.std(),
            'min_score': cv_scores.min(),
            'max_score': cv_scores.max()
        }
    
    def get_feature_importance(self, model_name, feature_names=None):
        """
        Get feature importance for a specific model.
        
        Args:
            model_name: Name of the model
            feature_names: List of feature names
            
        Returns:
            DataFrame with feature importance
        """
        if model_name not in self.feature_importance:
            raise ValueError(f"No feature importance available for model '{model_name}'")
        
        importance = self.feature_importance[model_name]
        
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(importance))]
        
        # Create DataFrame
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return feature_importance_df
    
    def predict_proba(self, model_name, X):
        """
        Get probability predictions from a specific model.
        
        Args:
            model_name: Name of the model
            X: Features to predict on
            
        Returns:
            Probability predictions
        """
        if model_name not in self.best_models:
            raise ValueError(f"Model '{model_name}' not found")
        
        model = self.best_models[model_name]
        return model.predict_proba(X)
    
    def save_models(self, filepath_prefix='credit_models'):
        """Save all trained models to disk."""
        for model_name, model in self.best_models.items():
            model_path = f"{filepath_prefix}_{model_name}.pkl"
            joblib.dump(model, model_path)
            print(f"Model {model_name} saved to {model_path}")
        
        # Save model scores
        scores_path = f"{filepath_prefix}_scores.pkl"
        joblib.dump(self.model_scores, scores_path)
        print(f"Model scores saved to {scores_path}")
        
        # Save feature importance
        if self.feature_importance:
            importance_path = f"{filepath_prefix}_importance.pkl"
            joblib.dump(self.feature_importance, importance_path)
            print(f"Feature importance saved to {importance_path}")
    
    def load_models(self, filepath_prefix='credit_models'):
        """Load trained models from disk."""
        import glob
        import os
        
        # Load models
        model_files = glob.glob(f"{filepath_prefix}_*.pkl")
        
        for model_file in model_files:
            if 'scores' not in model_file and 'importance' not in model_file:
                model_name = model_file.split('_')[-1].replace('.pkl', '')
                self.best_models[model_name] = joblib.load(model_file)
                print(f"Model {model_name} loaded from {model_file}")
        
        # Load scores
        scores_path = f"{filepath_prefix}_scores.pkl"
        if os.path.exists(scores_path):
            self.model_scores = joblib.load(scores_path)
            print(f"Model scores loaded from {scores_path}")
        
        # Load feature importance
        importance_path = f"{filepath_prefix}_importance.pkl"
        if os.path.exists(importance_path):
            self.feature_importance = joblib.load(importance_path)
            print(f"Feature importance loaded from {importance_path}")
    
    def get_model_summary(self):
        """Get a summary of all trained models and their performance."""
        if not self.model_scores:
            return "No models have been trained yet"
        
        summary = []
        summary.append("=" * 80)
        summary.append("CREDIT MODEL PERFORMANCE SUMMARY")
        summary.append("=" * 80)
        
        # Sort models by F1 score
        sorted_models = sorted(
            self.model_scores.items(),
            key=lambda x: x[1]['f1_score'],
            reverse=True
        )
        
        for i, (model_name, scores) in enumerate(sorted_models, 1):
            summary.append(f"\n{i}. {model_name.upper()}")
            summary.append("-" * 40)
            summary.append(f"Accuracy:  {scores['accuracy']:.4f}")
            summary.append(f"Precision: {scores['precision']:.4f}")
            summary.append(f"Recall:    {scores['recall']:.4f}")
            summary.append(f"F1-Score:  {scores['f1_score']:.4f}")
            summary.append(f"ROC-AUC:   {scores['roc_auc']:.4f}")
        
        summary.append("\n" + "=" * 80)
        return "\n".join(summary)

class CreditPredictor:
    """
    Simple interface for making credit predictions using trained models.
    """
    
    def __init__(self, model_trainer=None):
        self.model_trainer = model_trainer
        self.feature_names = None
    
    def set_feature_names(self, feature_names):
        """Set the feature names for the predictor."""
        self.feature_names = feature_names
    
    def predict_creditworthiness(self, features, model_name=None):
        """
        Predict creditworthiness for given features.
        
        Args:
            features: Dictionary or DataFrame with feature values
            model_name: Name of the model to use (if None, uses best model)
            
        Returns:
            Dictionary with prediction results
        """
        if self.model_trainer is None:
            raise ValueError("Model trainer not set")
        
        # Convert features to DataFrame if needed
        if isinstance(features, dict):
            features_df = pd.DataFrame([features])
        else:
            features_df = features.copy()
        
        # Ensure all required features are present
        if self.feature_names:
            missing_features = set(self.feature_names) - set(features_df.columns)
            if missing_features:
                raise ValueError(f"Missing features: {missing_features}")
            
            # Select only required features
            features_df = features_df[self.feature_names]
        
        # Get model to use
        if model_name is None:
            model_name, _, _ = self.model_trainer.get_best_model()
        
        # Make prediction
        probabilities = self.model_trainer.predict_proba(model_name, features_df)
        prediction = (probabilities[:, 1] > 0.5).astype(int)
        
        # Get confidence score
        confidence = np.max(probabilities, axis=1)
        
        return {
            'prediction': prediction[0],
            'probability_creditworthy': probabilities[0, 1],
            'probability_not_creditworthy': probabilities[0, 0],
            'confidence': confidence[0],
            'model_used': model_name
        }

if __name__ == "__main__":
    # Example usage
    from data_generator import CreditDataGenerator
    from feature_engineering import CreditFeatureEngineer
    
    # Generate and preprocess data
    generator = CreditDataGenerator()
    df = generator.generate_credit_data(n_samples=2000)
    
    engineer = CreditFeatureEngineer()
    df_processed, feature_names = engineer.preprocess_data(
        df, 
        target_column='creditworthy',
        handle_missing=True,
        create_derived_features=True,
        scale_features=True,
        select_features=True,
        n_features=20
    )
    
    # Prepare data for training
    X = df_processed.drop(columns=['creditworthy'])
    y = df_processed['creditworthy']
    
    # Train models
    trainer = CreditModelTrainer()
    scores = trainer.train_all_models(X, y)
    
    # Print summary
    print("\n" + trainer.get_model_summary())
    
    # Get best model
    best_name, best_model, best_score = trainer.get_best_model()
    print(f"\nBest model: {best_name} with F1 score: {best_score:.4f}")
    
    # Save models
    trainer.save_models()
    
    # Test prediction
    predictor = CreditPredictor(trainer)
    predictor.set_feature_names(feature_names)
    
    # Sample prediction
    sample_features = X.iloc[0].to_dict()
    prediction = predictor.predict_creditworthiness(sample_features)
    
    print(f"\nSample prediction:")
    print(f"Features: {sample_features}")
    print(f"Prediction: {'Creditworthy' if prediction['prediction'] == 1 else 'Not Creditworthy'}")
    print(f"Confidence: {prediction['confidence']:.4f}")
    print(f"Model used: {prediction['model_used']}")

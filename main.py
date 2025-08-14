#!/usr/bin/env python3
"""
Creditworthiness Prediction System
=================================

This script demonstrates the complete pipeline for creditworthiness prediction:
1. Data generation with realistic financial features
2. Feature engineering and preprocessing
3. Model training with multiple algorithms
4. Comprehensive model evaluation
5. Interactive predictions

Author: AI Assistant
Date: 2024
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

def main():
    """Main execution function for the creditworthiness prediction system."""
    
    print("=" * 80)
    print("CREDITWORTHINESS PREDICTION SYSTEM")
    print("=" * 80)
    print()
    
    try:
        # Import required modules
        print("Importing required modules...")
        from data_generator import CreditDataGenerator
        from feature_engineering import CreditFeatureEngineer
        from credit_models import CreditModelTrainer, CreditPredictor
        from model_evaluation import CreditModelEvaluator
        print("✓ All modules imported successfully")
        print()
        
        # Step 1: Generate synthetic credit data
        print("STEP 1: GENERATING SYNTHETIC CREDIT DATA")
        print("-" * 50)
        
        generator = CreditDataGenerator(random_state=42)
        print("Generating dataset with 5000 samples...")
        
        df = generator.generate_credit_data(
            n_samples=5000,
            n_features=20,
            n_informative=15,
            n_redundant=3,
            class_sep=1.2
        )
        
        print(f"✓ Dataset generated successfully")
        print(f"  - Shape: {df.shape}")
        print(f"  - Features: {len(df.columns) - 1}")
        print(f"  - Target distribution:")
        print(f"    * Creditworthy (1): {df['creditworthy'].sum()}")
        print(f"    * Not Creditworthy (0): {len(df) - df['creditworthy'].sum()}")
        print()
        
        # Step 2: Feature Engineering
        print("STEP 2: FEATURE ENGINEERING AND PREPROCESSING")
        print("-" * 50)
        
        engineer = CreditFeatureEngineer(random_state=42)
        print("Preprocessing data...")
        
        df_processed, feature_names = engineer.preprocess_data(
            df,
            target_column='creditworthy',
            handle_missing=True,
            create_derived_features=True,
            scale_features=True,
            select_features=True,
            n_features=25
        )
        
        print(f"✓ Data preprocessing completed")
        print(f"  - Original features: {len(df.columns) - 1}")
        print(f"  - Processed features: {len(feature_names)}")
        print(f"  - Final shape: {df_processed.shape}")
        print()
        
        # Step 3: Model Training
        print("STEP 3: TRAINING MACHINE LEARNING MODELS")
        print("-" * 50)
        
        # Prepare data for training
        X = df_processed.drop(columns=['creditworthy'])
        y = df_processed['creditworthy']
        
        trainer = CreditModelTrainer(random_state=42)
        print("Training multiple models with hyperparameter tuning...")
        print("This may take several minutes...")
        
        scores = trainer.train_all_models(X, y, test_size=0.2, cv_folds=5)
        
        print(f"✓ Model training completed")
        print(f"  - Models trained: {len(scores)}")
        print()
        
        # Step 4: Model Evaluation
        print("STEP 4: MODEL EVALUATION AND COMPARISON")
        print("-" * 50)
        
        # Get best model
        best_name, best_model, best_score = trainer.get_best_model()
        print(f"Best performing model: {best_name}")
        print(f"Best F1-Score: {best_score:.4f}")
        print()
        
        # Print model summary
        print("MODEL PERFORMANCE SUMMARY:")
        print(trainer.get_model_summary())
        print()
        
        # Step 5: Detailed Evaluation
        print("STEP 5: GENERATING EVALUATION VISUALIZATIONS")
        print("-" * 50)
        
        # Split data for evaluation
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        evaluator = CreditModelEvaluator(trainer)
        evaluation_results = evaluator.evaluate_all_models(X_test, y_test)
        
        print("✓ Model evaluation completed")
        print("Generating visualizations...")
        
        # Create output directory for plots
        os.makedirs('output', exist_ok=True)
        
        # Generate plots
        evaluator.plot_roc_curves(save_path='output/roc_curves.png')
        evaluator.plot_precision_recall_curves(save_path='output/precision_recall_curves.png')
        evaluator.plot_confusion_matrices(save_path='output/confusion_matrices.png')
        evaluator.plot_feature_importance_comparison(
            feature_names, 
            save_path='output/feature_importance.png'
        )
        evaluator.plot_model_performance_comparison(save_path='output/performance_comparison.png')
        
        # Create interactive dashboard
        evaluator.create_interactive_dashboard('output/credit_evaluation_dashboard.html')
        
        # Generate evaluation report
        report = evaluator.generate_evaluation_report('output/credit_evaluation_report.txt')
        
        print("✓ All visualizations and reports generated")
        print("  - Static plots saved to 'output/' directory")
        print("  - Interactive dashboard: output/credit_evaluation_dashboard.html")
        print("  - Evaluation report: output/credit_evaluation_report.txt")
        print()
        
        # Step 6: Interactive Predictions
        print("STEP 6: INTERACTIVE CREDIT PREDICTIONS")
        print("-" * 50)
        
        predictor = CreditPredictor(trainer)
        predictor.set_feature_names(feature_names)
        
        # Sample predictions
        print("Sample predictions using the best model:")
        print()
        
        # Get a few sample cases
        sample_indices = [0, 100, 500, 1000, 2000]
        
        for idx in sample_indices:
            if idx < len(X):
                sample_features = X.iloc[idx].to_dict()
                prediction = predictor.predict_creditworthiness(sample_features)
                
                status = "Creditworthy" if prediction['prediction'] == 1 else "Not Creditworthy"
                confidence = prediction['confidence']
                
                print(f"Sample {idx + 1}:")
                print(f"  Prediction: {status}")
                print(f"  Confidence: {confidence:.4f}")
                print(f"  Model used: {prediction['model_used']}")
                print()
        
        # Step 7: Save Models and Preprocessors
        print("STEP 7: SAVING MODELS AND PREPROCESSORS")
        print("-" * 50)
        
        # Save models
        trainer.save_models('output/credit_models')
        
        # Save preprocessors
        engineer.save_preprocessors('output/credit_preprocessors.pkl')
        
        print("✓ All models and preprocessors saved")
        print("  - Models: output/credit_models_*.pkl")
        print("  - Preprocessors: output/credit_preprocessors.pkl")
        print()
        
        # Step 8: Summary
        print("STEP 8: SYSTEM SUMMARY")
        print("-" * 50)
        
        print("🎉 CREDITWORTHINESS PREDICTION SYSTEM SUCCESSFULLY COMPLETED!")
        print()
        print("What was accomplished:")
        print("  ✓ Generated realistic synthetic credit data")
        print("  ✓ Implemented comprehensive feature engineering")
        print("  ✓ Trained 7 different ML models with hyperparameter tuning")
        print("  ✓ Evaluated models using multiple metrics (Accuracy, Precision, Recall, F1-Score, ROC-AUC)")
        print("  ✓ Generated detailed visualizations and reports")
        print("  ✓ Created interactive prediction interface")
        print("  ✓ Saved all models and preprocessors for future use")
        print()
        print("Key Features:")
        print("  • Multiple algorithms: Logistic Regression, Decision Trees, Random Forest, etc.")
        print("  • Advanced feature engineering with derived features")
        print("  • Comprehensive model evaluation with ROC curves and confusion matrices")
        print("  • Feature importance analysis")
        print("  • Interactive dashboard for model comparison")
        print("  • Production-ready model saving and loading")
        print()
        print("Next Steps:")
        print("  1. Review the generated visualizations in the 'output/' directory")
        print("  2. Open the interactive dashboard: output/credit_evaluation_dashboard.html")
        print("  3. Use the saved models for new predictions")
        print("  4. Customize the system for your specific credit data")
        print()
        print("=" * 80)
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("Please install required packages using: pip install -r requirements.txt")
        sys.exit(1)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Please check the error details above and try again.")
        sys.exit(1)

if __name__ == "__main__":
    main()

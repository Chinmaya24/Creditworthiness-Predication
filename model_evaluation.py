import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import (
    roc_curve, precision_recall_curve, confusion_matrix,
    classification_report, roc_auc_score, average_precision_score
)
from sklearn.model_selection import learning_curve
import warnings
warnings.filterwarnings('ignore')

class CreditModelEvaluator:
    """
    Comprehensive model evaluation and visualization for credit prediction models.
    Provides detailed analysis including ROC curves, confusion matrices, 
    feature importance plots, and statistical comparisons.
    """
    
    def __init__(self, model_trainer=None):
        self.model_trainer = model_trainer
        self.evaluation_results = {}
        
    def evaluate_all_models(self, X_test, y_test):
        """
        Evaluate all trained models on test data.
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary with evaluation results for all models
        """
        if self.model_trainer is None:
            raise ValueError("Model trainer not set")
        
        self.X_test = X_test
        self.y_test = y_test
        
        for model_name in self.model_trainer.best_models.keys():
            try:
                results = self.model_trainer.evaluate_model(model_name, X_test, y_test)
                self.evaluation_results[model_name] = results
                print(f"Evaluated {model_name}")
            except Exception as e:
                print(f"Error evaluating {model_name}: {str(e)}")
                continue
        
        return self.evaluation_results
    
    def plot_roc_curves(self, save_path=None):
        """
        Plot ROC curves for all models.
        """
        if not self.evaluation_results:
            raise ValueError("No evaluation results available. Run evaluate_all_models first.")

        plt.figure(figsize=(12, 8))

        y_test = self.y_test.astype(int)  # Ensure integer labels
        for name, model in self.model_trainer.best_models.items():
            # Use sklearn's roc_curve and plot manually
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            auc_score = roc_auc_score(y_test, y_pred_proba)
            plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.3f})')

        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', alpha=0.5)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()
    
    def plot_precision_recall_curves(self, figsize=(12, 8), save_path=None):
        """
        Plot Precision-Recall curves for all models.
        
        Args:
            figsize: Figure size
            save_path: Path to save the plot
        """
        if not self.evaluation_results:
            raise ValueError("No evaluation results available. Run evaluate_all_models first.")
        
        plt.figure(figsize=figsize)
        
        for model_name, results in self.evaluation_results.items():
            y_pred_proba = results['probabilities']
            precision, recall, _ = precision_recall_curve(self.y_test, y_pred_proba)
            ap_score = average_precision_score(self.y_test, y_pred_proba)
            plt.plot(recall, precision, label=f'{model_name} (AP = {ap_score:.3f})', linewidth=2)
        
        baseline = len(self.y_test[self.y_test == 1]) / len(self.y_test)
        plt.axhline(y=baseline, color='k', linestyle='--', alpha=0.5, label=f'Baseline (AP = {baseline:.3f})')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_confusion_matrices(self, figsize=(20, 15), save_path=None):
        """
        Plot confusion matrices for all models.
        
        Args:
            figsize: Figure size
            save_path: Path to save the plot
        """
        if not self.evaluation_results:
            raise ValueError("No evaluation results available. Run evaluate_all_models first.")
        
        n_models = len(self.evaluation_results)
        n_cols = 3
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        for idx, (model_name, results) in enumerate(self.evaluation_results.items()):
            row = idx // n_cols
            col = idx % n_cols
            
            cm = results['confusion_matrix']
            
            # Create heatmap
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Not Creditworthy', 'Creditworthy'],
                       yticklabels=['Not Creditworthy', 'Creditworthy'],
                       ax=axes[row, col])
            
            axes[row, col].set_title(f'{model_name}\nConfusion Matrix')
            axes[row, col].set_xlabel('Predicted')
            axes[row, col].set_ylabel('Actual')
        
        # Hide empty subplots
        for idx in range(n_models, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_feature_importance_comparison(self, feature_names=None, top_n=15, figsize=(15, 10), save_path=None):
        """
        Plot feature importance comparison across different models.
        
        Args:
            feature_names: List of feature names
            top_n: Number of top features to show
            figsize: Figure size
            save_path: Path to save the plot
        """
        if not self.model_trainer or not self.model_trainer.feature_importance:
            raise ValueError("No feature importance data available")
        
        # Get feature importance for all models
        importance_data = {}
        for model_name, importance in self.model_trainer.feature_importance.items():
            if feature_names is None:
                feature_names = [f'feature_{i}' for i in range(len(importance))]
            
            # Create DataFrame and get top features
            df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False).head(top_n)
            
            importance_data[model_name] = df
        
        # Create comparison plot
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.ravel()
        
        for idx, (model_name, df) in enumerate(importance_data.items()):
            if idx < 4:  # Limit to 4 subplots
                sns.barplot(data=df, x='importance', y='feature', ax=axes[idx])
                axes[idx].set_title(f'{model_name} - Top {top_n} Features')
                axes[idx].set_xlabel('Importance')
        
        # Hide empty subplots
        for idx in range(len(importance_data), 4):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_model_performance_comparison(self, metrics=['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc'], 
                                        figsize=(15, 10), save_path=None):
        """
        Plot performance comparison across different metrics.
        
        Args:
            metrics: List of metrics to compare
            figsize: Figure size
            save_path: Path to save the plot
        """
        if not self.evaluation_results:
            raise ValueError("No evaluation results available. Run evaluate_all_models first.")
        
        # Prepare data for plotting
        comparison_data = []
        for model_name, results in self.evaluation_results.items():
            for metric in metrics:
                if metric in results['metrics']:
                    comparison_data.append({
                        'Model': model_name.replace('_', ' ').title(),
                        'Metric': metric.replace('_', ' ').title(),
                        'Score': results['metrics'][metric]
                    })
        
        df_comparison = pd.DataFrame(comparison_data)
        
        # Create subplots for each metric
        n_metrics = len(metrics)
        n_cols = 2
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        for idx, metric in enumerate(metrics):
            row = idx // n_cols
            col = idx % n_cols
            
            metric_data = df_comparison[df_comparison['Metric'] == metric.replace('_', ' ').title()]
            
            if not metric_data.empty:
                sns.barplot(data=metric_data, x='Score', y='Model', ax=axes[row, col])
                axes[row, col].set_title(f'{metric.replace("_", " ").title()} Comparison')
                axes[row, col].set_xlabel('Score')
        
        # Hide empty subplots
        for idx in range(n_metrics, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def create_interactive_dashboard(self, save_path=None):
        """
        Create an interactive Plotly dashboard with all evaluation metrics.
        
        Args:
            save_path: Path to save the HTML dashboard
        """
        if not self.evaluation_results:
            raise ValueError("No evaluation results available. Run evaluate_all_models first.")
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('ROC Curves', 'Precision-Recall Curves', 
                          'Accuracy Comparison', 'F1-Score Comparison',
                          'Feature Importance (Top Model)', 'Model Performance Heatmap'),
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "heatmap"}]]
        )
        
        # ROC Curves
        for model_name, results in self.evaluation_results.items():
            y_pred_proba = results['probabilities']
            fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
            auc_score = roc_auc_score(self.y_test, y_pred_proba)
            fig.add_trace(
                go.Scatter(x=fpr, y=tpr, name=f'{model_name} (AUC: {auc_score:.3f})',
                          mode='lines', line=dict(width=2)),
                row=1, col=1
            )
        
        # Add diagonal line
        fig.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1], name='Random Classifier',
                      mode='lines', line=dict(dash='dash', color='black')),
            row=1, col=1
        )
        
        # Precision-Recall Curves
        for model_name, results in self.evaluation_results.items():
            y_pred_proba = results['probabilities']
            precision, recall, _ = precision_recall_curve(self.y_test, y_pred_proba)
            ap_score = average_precision_score(self.y_test, y_pred_proba)
            fig.add_trace(
                go.Scatter(x=recall, y=precision, name=f'{model_name} (AP: {ap_score:.3f})',
                          mode='lines', line=dict(width=2)),
                row=1, col=2
            )
        
        # Accuracy Comparison
        model_names = list(self.evaluation_results.keys())
        accuracy_scores = [self.evaluation_results[name]['metrics']['accuracy'] for name in model_names]
        
        fig.add_trace(
            go.Bar(x=[name.replace('_', ' ').title() for name in model_names], 
                  y=accuracy_scores, name='Accuracy', marker_color='lightblue'),
            row=2, col=1
        )
        
        # F1-Score Comparison
        f1_scores = [self.evaluation_results[name]['metrics']['f1_score'] for name in model_names]
        
        fig.add_trace(
            go.Bar(x=[name.replace('_', ' ').title() for name in model_names], 
                  y=f1_scores, name='F1-Score', marker_color='lightgreen'),
            row=2, col=2
        )
        
        # Feature Importance (Top Model)
        if self.model_trainer and self.model_trainer.feature_importance:
            best_model_name, _, _ = self.model_trainer.get_best_model()
            if best_model_name in self.model_trainer.feature_importance:
                importance = self.model_trainer.feature_importance[best_model_name]
                feature_names = [f'feature_{i}' for i in range(len(importance))]
                
                # Get top 10 features
                top_indices = np.argsort(importance)[-10:]
                
                fig.add_trace(
                    go.Bar(x=[feature_names[i] for i in top_indices], 
                          y=[importance[i] for i in top_indices], 
                          name='Feature Importance', marker_color='orange'),
                    row=3, col=1
                )
        
        # Model Performance Heatmap
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        heatmap_data = []
        
        for model_name in model_names:
            row_data = []
            for metric in metrics:
                row_data.append(self.evaluation_results[model_name]['metrics'][metric])
            heatmap_data.append(row_data)
        
        fig.add_trace(
            go.Heatmap(z=heatmap_data, 
                      x=[m.replace('_', ' ').title() for m in metrics],
                      y=[name.replace('_', ' ').title() for name in model_names],
                      colorscale='Viridis', name='Performance Heatmap'),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(
            title='Credit Model Evaluation Dashboard',
            height=1200,
            showlegend=True
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="False Positive Rate", row=1, col=1)
        fig.update_yaxes(title_text="True Positive Rate", row=1, col=1)
        fig.update_xaxes(title_text="Recall", row=1, col=2)
        fig.update_yaxes(title_text="Precision", row=1, col=2)
        fig.update_xaxes(title_text="Models", row=2, col=1)
        fig.update_yaxes(title_text="Score", row=2, col=1)
        fig.update_xaxes(title_text="Models", row=2, col=2)
        fig.update_yaxes(title_text="Score", row=2, col=2)
        fig.update_xaxes(title_text="Features", row=3, col=1)
        fig.update_yaxes(title_text="Importance", row=3, col=1)
        fig.update_xaxes(title_text="Metrics", row=3, col=2)
        fig.update_yaxes(title_text="Models", row=3, col=2)
        
        if save_path:
            fig.write_html(save_path)
        
        fig.show()
    
    def generate_evaluation_report(self, save_path=None):
        """
        Generate a comprehensive evaluation report.
        
        Args:
            save_path: Path to save the report
            
        Returns:
            String containing the report
        """
        if not self.evaluation_results:
            raise ValueError("No evaluation results available. Run evaluate_all_models first.")
        
        report = []
        report.append("=" * 80)
        report.append("CREDIT MODEL EVALUATION REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Model Performance Summary
        report.append("MODEL PERFORMANCE SUMMARY")
        report.append("-" * 40)
        
        summary_data = []
        for model_name, results in self.evaluation_results.items():
            metrics = results['metrics']
            summary_data.append({
                'Model': model_name,
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1-Score': f"{metrics['f1_score']:.4f}",
                'ROC-AUC': f"{metrics['roc_auc']:.4f}"
            })
        
        summary_df = pd.DataFrame(summary_data)
        report.append(summary_df.to_string(index=False))
        report.append("")
        
        # Detailed Results for Each Model
        for model_name, results in self.evaluation_results.items():
            report.append(f"DETAILED RESULTS: {model_name.upper()}")
            report.append("-" * 50)
            
            metrics = results['metrics']
            report.append(f"Accuracy:  {metrics['accuracy']:.4f}")
            report.append(f"Precision: {metrics['precision']:.4f}")
            report.append(f"Recall:    {metrics['recall']:.4f}")
            report.append(f"F1-Score:  {metrics['f1_score']:.4f}")
            report.append(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
            report.append("")
            
            # Confusion Matrix
            cm = results['confusion_matrix']
            report.append("Confusion Matrix:")
            report.append("                Predicted")
            report.append("Actual    0 (Not Creditworthy)    1 (Creditworthy)")
            report.append(f"0 (Not Creditworthy)    {cm[0,0]:>8}    {cm[0,1]:>8}")
            report.append(f"1 (Creditworthy)        {cm[1,0]:>8}    {cm[1,1]:>8}")
            report.append("")
            
            # Classification Report
            report.append("Classification Report:")
            # Convert dict to string if needed
            if isinstance(results['classification_report'], dict):
                report.append(str(results['classification_report']))
            else:
                report.append(results['classification_report'])
            report.append("")
        
        # Feature Importance Summary
        if self.model_trainer and self.model_trainer.feature_importance:
            report.append("FEATURE IMPORTANCE SUMMARY")
            report.append("-" * 40)
            
            for model_name, importance in self.model_trainer.feature_importance.items():
                report.append(f"\n{model_name.upper()}:")
                top_features = np.argsort(importance)[-5:]  # Top 5 features
                for i, idx in enumerate(reversed(top_features)):
                    report.append(f"  {i+1}. Feature {idx}: {importance[idx]:.4f}")
        
        report.append("\n" + "=" * 80)
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            print(f"Evaluation report saved to {save_path}")
        
        return report_text

if __name__ == "__main__":
    # Example usage
    from data_generator import CreditDataGenerator
    from feature_engineering import CreditFeatureEngineer
    from credit_models import CreditModelTrainer
    
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
    
    # Initialize evaluator
    evaluator = CreditModelEvaluator(trainer)
    
    # Split data for evaluation
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Evaluate all models
    evaluation_results = evaluator.evaluate_all_models(X_test, y_test)
    
    # Generate plots
    evaluator.plot_roc_curves()
    evaluator.plot_precision_recall_curves()
    evaluator.plot_confusion_matrices()
    evaluator.plot_feature_importance_comparison(feature_names)
    evaluator.plot_model_performance_comparison()
    
    # Create interactive dashboard
    evaluator.create_interactive_dashboard('credit_evaluation_dashboard.html')
    
    # Generate report
    report = evaluator.generate_evaluation_report('credit_evaluation_report.txt')
    print("Evaluation completed successfully!")

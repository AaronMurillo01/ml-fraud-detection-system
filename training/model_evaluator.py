#!/usr/bin/env python3
"""Model evaluation utilities for fraud detection system.

This module provides comprehensive model evaluation capabilities including:
- Performance metrics calculation (AUC, precision, recall, F1)
- Model comparison and ranking
- Threshold optimization
- Visualization of model performance
- Business impact analysis
"""

import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, log_loss,
    confusion_matrix, classification_report,
    precision_recall_curve, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Import project modules
sys.path.append(str(Path(__file__).parent.parent))
from service.ml_service import FraudDetectionModel, ModelEnsemble

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Container for model evaluation metrics."""
    
    # Classification metrics
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc: float
    average_precision: float
    log_loss: float
    
    # Confusion matrix components
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int
    
    # Business metrics
    precision_at_k: Optional[Dict[int, float]] = None
    recall_at_k: Optional[Dict[int, float]] = None
    
    # Threshold analysis
    optimal_threshold: Optional[float] = None
    threshold_metrics: Optional[Dict[str, List[float]]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1': self.f1_score,
            'auc': self.auc,
            'average_precision': self.average_precision,
            'log_loss': self.log_loss,
            'true_positives': self.true_positives,
            'true_negatives': self.true_negatives,
            'false_positives': self.false_positives,
            'false_negatives': self.false_negatives,
            'optimal_threshold': self.optimal_threshold
        }


class ModelEvaluator:
    """Comprehensive model evaluation utilities."""
    
    def __init__(self, default_threshold: float = 0.5):
        """Initialize ModelEvaluator.
        
        Args:
            default_threshold: Default classification threshold
        """
        self.default_threshold = default_threshold
    
    def evaluate_model(self, model: FraudDetectionModel, 
                      X_test: pd.DataFrame, y_test: pd.Series,
                      threshold: Optional[float] = None) -> Dict[str, float]:
        """Evaluate a single model comprehensively.
        
        Args:
            model: Trained fraud detection model
            X_test: Test features
            y_test: Test labels
            threshold: Classification threshold (if None, uses optimal)
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"Evaluating model: {model.model_name}")
        
        # Get predictions
        y_pred_proba = model.predict_proba(X_test)
        
        # Find optimal threshold if not provided
        if threshold is None:
            threshold = self._find_optimal_threshold(y_test, y_pred_proba)
        
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Calculate metrics
        metrics = self._calculate_comprehensive_metrics(
            y_test, y_pred, y_pred_proba, threshold
        )
        
        return metrics.to_dict()
    
    def evaluate_ensemble(self, ensemble: ModelEnsemble,
                         X_test: pd.DataFrame, y_test: pd.Series,
                         threshold: Optional[float] = None) -> Dict[str, float]:
        """Evaluate ensemble model.
        
        Args:
            ensemble: Trained ensemble model
            X_test: Test features
            y_test: Test labels
            threshold: Classification threshold
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Evaluating ensemble model")
        
        # Get ensemble predictions
        y_pred_proba = ensemble.predict_proba(X_test)
        
        # Find optimal threshold if not provided
        if threshold is None:
            threshold = self._find_optimal_threshold(y_test, y_pred_proba)
        
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Calculate metrics
        metrics = self._calculate_comprehensive_metrics(
            y_test, y_pred, y_pred_proba, threshold
        )
        
        return metrics.to_dict()
    
    def _calculate_comprehensive_metrics(self, y_true: pd.Series, 
                                       y_pred: np.ndarray,
                                       y_pred_proba: np.ndarray,
                                       threshold: float) -> EvaluationMetrics:
        """Calculate comprehensive evaluation metrics."""
        # Basic classification metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # Probabilistic metrics
        auc = roc_auc_score(y_true, y_pred_proba)
        avg_precision = average_precision_score(y_true, y_pred_proba)
        
        # Log loss (with clipping to avoid numerical issues)
        y_pred_proba_clipped = np.clip(y_pred_proba, 1e-15, 1 - 1e-15)
        logloss = log_loss(y_true, y_pred_proba_clipped)
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Precision/Recall at K
        precision_at_k = self._calculate_precision_at_k(y_true, y_pred_proba)
        recall_at_k = self._calculate_recall_at_k(y_true, y_pred_proba)
        
        # Threshold analysis
        threshold_metrics = self._analyze_thresholds(y_true, y_pred_proba)
        
        return EvaluationMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            auc=auc,
            average_precision=avg_precision,
            log_loss=logloss,
            true_positives=int(tp),
            true_negatives=int(tn),
            false_positives=int(fp),
            false_negatives=int(fn),
            precision_at_k=precision_at_k,
            recall_at_k=recall_at_k,
            optimal_threshold=threshold,
            threshold_metrics=threshold_metrics
        )
    
    def _find_optimal_threshold(self, y_true: pd.Series, 
                               y_pred_proba: np.ndarray,
                               metric: str = 'f1') -> float:
        """Find optimal classification threshold.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            metric: Metric to optimize ('f1', 'precision', 'recall', 'youden')
            
        Returns:
            Optimal threshold
        """
        thresholds = np.linspace(0.01, 0.99, 99)
        scores = []
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            if metric == 'f1':
                score = f1_score(y_true, y_pred, zero_division=0)
            elif metric == 'precision':
                score = precision_score(y_true, y_pred, zero_division=0)
            elif metric == 'recall':
                score = recall_score(y_true, y_pred, zero_division=0)
            elif metric == 'youden':
                # Youden's J statistic (sensitivity + specificity - 1)
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                score = sensitivity + specificity - 1
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            scores.append(score)
        
        optimal_idx = np.argmax(scores)
        optimal_threshold = thresholds[optimal_idx]
        
        logger.info(f"Optimal threshold ({metric}): {optimal_threshold:.3f}")
        return optimal_threshold
    
    def _calculate_precision_at_k(self, y_true: pd.Series, 
                                 y_pred_proba: np.ndarray,
                                 k_values: List[int] = None) -> Dict[int, float]:
        """Calculate precision at top K predictions."""
        if k_values is None:
            k_values = [10, 50, 100, 500, 1000]
        
        # Sort by prediction probability (descending)
        sorted_indices = np.argsort(y_pred_proba)[::-1]
        y_true_sorted = y_true.iloc[sorted_indices]
        
        precision_at_k = {}
        
        for k in k_values:
            if k <= len(y_true_sorted):
                top_k_true = y_true_sorted.iloc[:k]
                precision_at_k[k] = top_k_true.sum() / k
            else:
                precision_at_k[k] = None
        
        return precision_at_k
    
    def _calculate_recall_at_k(self, y_true: pd.Series, 
                              y_pred_proba: np.ndarray,
                              k_values: List[int] = None) -> Dict[int, float]:
        """Calculate recall at top K predictions."""
        if k_values is None:
            k_values = [10, 50, 100, 500, 1000]
        
        total_positives = y_true.sum()
        if total_positives == 0:
            return {k: 0.0 for k in k_values}
        
        # Sort by prediction probability (descending)
        sorted_indices = np.argsort(y_pred_proba)[::-1]
        y_true_sorted = y_true.iloc[sorted_indices]
        
        recall_at_k = {}
        
        for k in k_values:
            if k <= len(y_true_sorted):
                top_k_true = y_true_sorted.iloc[:k]
                recall_at_k[k] = top_k_true.sum() / total_positives
            else:
                recall_at_k[k] = y_true_sorted.sum() / total_positives
        
        return recall_at_k
    
    def _analyze_thresholds(self, y_true: pd.Series, 
                           y_pred_proba: np.ndarray) -> Dict[str, List[float]]:
        """Analyze performance across different thresholds."""
        thresholds = np.linspace(0.01, 0.99, 50)
        
        precisions = []
        recalls = []
        f1_scores = []
        accuracies = []
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            precisions.append(precision_score(y_true, y_pred, zero_division=0))
            recalls.append(recall_score(y_true, y_pred, zero_division=0))
            f1_scores.append(f1_score(y_true, y_pred, zero_division=0))
            accuracies.append(accuracy_score(y_true, y_pred))
        
        return {
            'thresholds': thresholds.tolist(),
            'precisions': precisions,
            'recalls': recalls,
            'f1_scores': f1_scores,
            'accuracies': accuracies
        }
    
    def compare_models(self, models: List[FraudDetectionModel],
                      X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
        """Compare multiple models side by side.
        
        Args:
            models: List of trained models
            X_test: Test features
            y_test: Test labels
            
        Returns:
            DataFrame with comparison results
        """
        logger.info(f"Comparing {len(models)} models")
        
        results = []
        
        for model in models:
            try:
                metrics = self.evaluate_model(model, X_test, y_test)
                metrics['model_name'] = model.model_name
                results.append(metrics)
            except Exception as e:
                logger.error(f"Error evaluating {model.model_name}: {e}")
                continue
        
        if not results:
            raise ValueError("No models could be evaluated")
        
        comparison_df = pd.DataFrame(results)
        
        # Sort by AUC (descending)
        comparison_df = comparison_df.sort_values('auc', ascending=False)
        
        return comparison_df
    
    def calculate_business_impact(self, y_true: pd.Series, y_pred: np.ndarray,
                                 transaction_amounts: pd.Series,
                                 investigation_cost: float = 10.0,
                                 fraud_recovery_rate: float = 0.3) -> Dict[str, float]:
        """Calculate business impact metrics.
        
        Args:
            y_true: True fraud labels
            y_pred: Predicted fraud labels
            transaction_amounts: Transaction amounts
            investigation_cost: Cost per investigation
            fraud_recovery_rate: Rate of fraud amount recovery
            
        Returns:
            Dictionary with business impact metrics
        """
        # Confusion matrix components
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Calculate costs and savings
        investigation_costs = (tp + fp) * investigation_cost
        
        # Fraud amounts
        fraud_mask = y_true == 1
        total_fraud_amount = transaction_amounts[fraud_mask].sum()
        
        # Detected fraud amounts
        detected_fraud_mask = (y_true == 1) & (y_pred == 1)
        detected_fraud_amount = transaction_amounts[detected_fraud_mask].sum()
        
        # Missed fraud amounts
        missed_fraud_mask = (y_true == 1) & (y_pred == 0)
        missed_fraud_amount = transaction_amounts[missed_fraud_mask].sum()
        
        # False positive amounts (legitimate transactions flagged)
        false_positive_mask = (y_true == 0) & (y_pred == 1)
        false_positive_amount = transaction_amounts[false_positive_mask].sum()
        
        # Calculate savings and losses
        fraud_savings = detected_fraud_amount * fraud_recovery_rate
        fraud_losses = missed_fraud_amount
        
        # Net benefit
        net_benefit = fraud_savings - investigation_costs
        
        # ROI calculation
        roi = (fraud_savings - investigation_costs) / investigation_costs if investigation_costs > 0 else 0
        
        return {
            'total_fraud_amount': float(total_fraud_amount),
            'detected_fraud_amount': float(detected_fraud_amount),
            'missed_fraud_amount': float(missed_fraud_amount),
            'false_positive_amount': float(false_positive_amount),
            'investigation_costs': float(investigation_costs),
            'fraud_savings': float(fraud_savings),
            'fraud_losses': float(fraud_losses),
            'net_benefit': float(net_benefit),
            'roi': float(roi),
            'fraud_detection_rate': float(detected_fraud_amount / total_fraud_amount) if total_fraud_amount > 0 else 0,
            'precision_weighted': float(detected_fraud_amount / (detected_fraud_amount + false_positive_amount)) if (detected_fraud_amount + false_positive_amount) > 0 else 0
        }
    
    def plot_roc_curve(self, y_true: pd.Series, y_pred_proba: np.ndarray,
                      model_name: str = "Model", save_path: Optional[str] = None) -> None:
        """Plot ROC curve."""
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        auc_score = roc_auc_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, linewidth=2, label=f'{model_name} (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_precision_recall_curve(self, y_true: pd.Series, y_pred_proba: np.ndarray,
                                   model_name: str = "Model", save_path: Optional[str] = None) -> None:
        """Plot Precision-Recall curve."""
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        avg_precision = average_precision_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, linewidth=2, 
                label=f'{model_name} (AP = {avg_precision:.3f})')
        
        # Baseline (random classifier)
        baseline = y_true.mean()
        plt.axhline(y=baseline, color='k', linestyle='--', linewidth=1, 
                   label=f'Baseline (AP = {baseline:.3f})')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_confusion_matrix(self, y_true: pd.Series, y_pred: np.ndarray,
                             model_name: str = "Model", save_path: Optional[str] = None) -> None:
        """Plot confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Legitimate', 'Fraud'],
                   yticklabels=['Legitimate', 'Fraud'])
        
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def generate_evaluation_report(self, model: FraudDetectionModel,
                                  X_test: pd.DataFrame, y_test: pd.Series,
                                  transaction_amounts: Optional[pd.Series] = None) -> str:
        """Generate comprehensive evaluation report.
        
        Args:
            model: Trained model to evaluate
            X_test: Test features
            y_test: Test labels
            transaction_amounts: Transaction amounts for business impact
            
        Returns:
            Formatted evaluation report string
        """
        # Evaluate model
        metrics = self.evaluate_model(model, X_test, y_test)
        
        # Get predictions for business impact
        y_pred_proba = model.predict_proba(X_test)
        y_pred = (y_pred_proba >= metrics['optimal_threshold']).astype(int)
        
        report = f"""
{'='*60}
MODEL EVALUATION REPORT
{'='*60}

Model: {model.model_name}
Model Version: {model.model_version}
Evaluation Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

{'='*60}
PERFORMANCE METRICS
{'='*60}

Classification Metrics:
  • Accuracy:           {metrics['accuracy']:.4f}
  • Precision:          {metrics['precision']:.4f}
  • Recall:             {metrics['recall']:.4f}
  • F1-Score:           {metrics['f1']:.4f}
  • AUC-ROC:            {metrics['auc']:.4f}
  • Average Precision:  {metrics['average_precision']:.4f}
  • Log Loss:           {metrics['log_loss']:.4f}

Confusion Matrix:
  • True Positives:     {metrics['true_positives']:,}
  • True Negatives:     {metrics['true_negatives']:,}
  • False Positives:    {metrics['false_positives']:,}
  • False Negatives:    {metrics['false_negatives']:,}

Optimal Threshold:      {metrics['optimal_threshold']:.4f}
"""
        
        # Add business impact if transaction amounts provided
        if transaction_amounts is not None:
            business_metrics = self.calculate_business_impact(
                y_test, y_pred, transaction_amounts
            )
            
            report += f"""
{'='*60}
BUSINESS IMPACT ANALYSIS
{'='*60}

Fraud Detection:
  • Total Fraud Amount:     ${business_metrics['total_fraud_amount']:,.2f}
  • Detected Fraud Amount:  ${business_metrics['detected_fraud_amount']:,.2f}
  • Missed Fraud Amount:    ${business_metrics['missed_fraud_amount']:,.2f}
  • Detection Rate:         {business_metrics['fraud_detection_rate']:.2%}

Costs and Savings:
  • Investigation Costs:    ${business_metrics['investigation_costs']:,.2f}
  • Fraud Savings:          ${business_metrics['fraud_savings']:,.2f}
  • Net Benefit:            ${business_metrics['net_benefit']:,.2f}
  • ROI:                    {business_metrics['roi']:.2%}

False Positives:
  • FP Transaction Amount:  ${business_metrics['false_positive_amount']:,.2f}
  • Precision (Weighted):   {business_metrics['precision_weighted']:.4f}
"""
        
        report += f"""
{'='*60}
RECOMMendations
{'='*60}
"""
        
        # Add recommendations based on metrics
        if metrics['precision'] < 0.5:
            report += "\n• Consider increasing classification threshold to reduce false positives"
        
        if metrics['recall'] < 0.7:
            report += "\n• Consider lowering classification threshold to catch more fraud"
        
        if metrics['auc'] < 0.8:
            report += "\n• Model performance is below optimal - consider feature engineering or model tuning"
        
        if metrics['auc'] > 0.95:
            report += "\n• Excellent model performance - ready for production deployment"
        
        report += "\n\n" + "="*60 + "\n"
        
        return report


def main():
    """Example usage of ModelEvaluator."""
    # This would typically be called with real models and data
    print("ModelEvaluator module loaded successfully")
    print("Use this module to evaluate trained fraud detection models")


if __name__ == "__main__":
    main()
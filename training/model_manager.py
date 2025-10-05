#!/usr/bin/env python3
"""Model management utilities for fraud detection system.

This module provides comprehensive model lifecycle management including:
- Model versioning and metadata tracking
- Model storage and retrieval
- Model deployment and rollback
- Performance monitoring and alerting
- A/B testing support
"""

import os
import sys
import json
import pickle
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# Import project modules
sys.path.append(str(Path(__file__).parent.parent))
from service.ml_service import FraudDetectionModel, ModelEnsemble
from training.model_evaluator import ModelEvaluator, EvaluationMetrics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    """Model metadata container."""
    
    model_id: str
    model_name: str
    model_version: str
    model_type: str
    created_at: datetime
    created_by: str
    
    # Training information
    training_data_hash: str
    training_samples: int
    training_features: List[str]
    hyperparameters: Dict[str, Any]
    
    # Performance metrics
    validation_metrics: Dict[str, float]
    test_metrics: Optional[Dict[str, float]] = None
    
    # Deployment information
    deployment_status: str = "trained"  # trained, staging, production, retired
    deployment_date: Optional[datetime] = None
    
    # Model artifacts
    model_path: str = ""
    mlflow_run_id: Optional[str] = None
    
    # Business metrics
    business_impact: Optional[Dict[str, float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with datetime serialization."""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        if self.deployment_date:
            data['deployment_date'] = self.deployment_date.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetadata':
        """Create from dictionary with datetime deserialization."""
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        if data.get('deployment_date'):
            data['deployment_date'] = datetime.fromisoformat(data['deployment_date'])
        return cls(**data)


class ModelRegistry:
    """Model registry for tracking and managing models."""
    
    def __init__(self, registry_path: str = "models/registry"):
        """Initialize ModelRegistry.
        
        Args:
            registry_path: Path to store model registry
        """
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        
        self.metadata_file = self.registry_path / "model_metadata.json"
        self.models_metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict[str, ModelMetadata]:
        """Load model metadata from file."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    data = json.load(f)
                return {k: ModelMetadata.from_dict(v) for k, v in data.items()}
            except Exception as e:
                logger.error(f"Error loading metadata: {e}")
                return {}
        return {}
    
    def _save_metadata(self) -> None:
        """Save model metadata to file."""
        try:
            data = {k: v.to_dict() for k, v in self.models_metadata.items()}
            with open(self.metadata_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")
    
    def register_model(self, model: FraudDetectionModel, 
                      metadata: ModelMetadata) -> str:
        """Register a new model.
        
        Args:
            model: Trained model to register
            metadata: Model metadata
            
        Returns:
            Model ID
        """
        model_id = metadata.model_id
        
        # Save model
        model_path = self.registry_path / f"{model_id}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Update metadata with path
        metadata.model_path = str(model_path)
        
        # Store metadata
        self.models_metadata[model_id] = metadata
        self._save_metadata()
        
        logger.info(f"Registered model: {model_id}")
        return model_id
    
    def get_model(self, model_id: str) -> Optional[FraudDetectionModel]:
        """Retrieve model by ID.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Loaded model or None if not found
        """
        if model_id not in self.models_metadata:
            logger.warning(f"Model not found: {model_id}")
            return None
        
        metadata = self.models_metadata[model_id]
        
        try:
            with open(metadata.model_path, 'rb') as f:
                model = pickle.load(f)
            return model
        except Exception as e:
            logger.error(f"Error loading model {model_id}: {e}")
            return None
    
    def get_metadata(self, model_id: str) -> Optional[ModelMetadata]:
        """Get model metadata.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Model metadata or None if not found
        """
        return self.models_metadata.get(model_id)
    
    def list_models(self, status: Optional[str] = None) -> List[ModelMetadata]:
        """List all models with optional status filter.
        
        Args:
            status: Filter by deployment status
            
        Returns:
            List of model metadata
        """
        models = list(self.models_metadata.values())
        
        if status:
            models = [m for m in models if m.deployment_status == status]
        
        # Sort by creation date (newest first)
        models.sort(key=lambda x: x.created_at, reverse=True)
        
        return models
    
    def update_deployment_status(self, model_id: str, status: str) -> bool:
        """Update model deployment status.
        
        Args:
            model_id: Model identifier
            status: New deployment status
            
        Returns:
            True if successful, False otherwise
        """
        if model_id not in self.models_metadata:
            logger.warning(f"Model not found: {model_id}")
            return False
        
        self.models_metadata[model_id].deployment_status = status
        if status in ['staging', 'production']:
            self.models_metadata[model_id].deployment_date = datetime.now()
        
        self._save_metadata()
        logger.info(f"Updated {model_id} status to {status}")
        return True
    
    def delete_model(self, model_id: str) -> bool:
        """Delete model and its metadata.
        
        Args:
            model_id: Model identifier
            
        Returns:
            True if successful, False otherwise
        """
        if model_id not in self.models_metadata:
            logger.warning(f"Model not found: {model_id}")
            return False
        
        metadata = self.models_metadata[model_id]
        
        # Delete model file
        try:
            if os.path.exists(metadata.model_path):
                os.remove(metadata.model_path)
        except Exception as e:
            logger.error(f"Error deleting model file: {e}")
        
        # Remove from metadata
        del self.models_metadata[model_id]
        self._save_metadata()
        
        logger.info(f"Deleted model: {model_id}")
        return True


class ModelManager:
    """Comprehensive model lifecycle management."""
    
    def __init__(self, registry_path: str = "models/registry",
                 mlflow_tracking_uri: Optional[str] = None):
        """Initialize ModelManager.
        
        Args:
            registry_path: Path to model registry
            mlflow_tracking_uri: MLflow tracking server URI
        """
        self.registry = ModelRegistry(registry_path)
        self.evaluator = ModelEvaluator()
        
        # Initialize MLflow
        if mlflow_tracking_uri:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
        
        self.mlflow_client = MlflowClient()
    
    def train_and_register_model(self, model_name: str, model_type: str,
                                model: BaseEstimator, X_train: pd.DataFrame,
                                y_train: pd.Series, X_val: pd.DataFrame,
                                y_val: pd.Series, hyperparameters: Dict[str, Any],
                                created_by: str = "system") -> str:
        """Train and register a new model.
        
        Args:
            model_name: Name of the model
            model_type: Type of model (e.g., 'xgboost', 'lightgbm')
            model: Sklearn-compatible model
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            hyperparameters: Model hyperparameters
            created_by: User who created the model
            
        Returns:
            Model ID
        """
        logger.info(f"Training and registering model: {model_name}")
        
        # Start MLflow run
        with mlflow.start_run() as run:
            # Train model
            model.fit(X_train, y_train)
            
            # Create FraudDetectionModel wrapper
            fraud_model = FraudDetectionModel(
                model=model,
                model_name=model_name,
                model_version="1.0.0",
                feature_names=list(X_train.columns)
            )
            
            # Evaluate on validation set
            val_metrics = self.evaluator.evaluate_model(fraud_model, X_val, y_val)
            
            # Log to MLflow
            mlflow.log_params(hyperparameters)
            mlflow.log_metrics(val_metrics)
            mlflow.sklearn.log_model(model, "model")
            
            # Create metadata
            model_id = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            metadata = ModelMetadata(
                model_id=model_id,
                model_name=model_name,
                model_version="1.0.0",
                model_type=model_type,
                created_at=datetime.now(),
                created_by=created_by,
                training_data_hash=self._calculate_data_hash(X_train, y_train),
                training_samples=len(X_train),
                training_features=list(X_train.columns),
                hyperparameters=hyperparameters,
                validation_metrics=val_metrics,
                mlflow_run_id=run.info.run_id
            )
            
            # Register model
            self.registry.register_model(fraud_model, metadata)
            
            logger.info(f"Model registered with ID: {model_id}")
            return model_id
    
    def deploy_model(self, model_id: str, environment: str = "staging") -> bool:
        """Deploy model to specified environment.
        
        Args:
            model_id: Model identifier
            environment: Target environment ('staging' or 'production')
            
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Deploying model {model_id} to {environment}")
        
        # Validate model exists
        model = self.registry.get_model(model_id)
        if not model:
            logger.error(f"Model not found: {model_id}")
            return False
        
        # Update deployment status
        success = self.registry.update_deployment_status(model_id, environment)
        
        if success:
            logger.info(f"Model {model_id} deployed to {environment}")
        
        return success
    
    def rollback_model(self, current_model_id: str, 
                      previous_model_id: str) -> bool:
        """Rollback from current model to previous model.
        
        Args:
            current_model_id: Current production model ID
            previous_model_id: Previous model ID to rollback to
            
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Rolling back from {current_model_id} to {previous_model_id}")
        
        # Validate both models exist
        current_model = self.registry.get_model(current_model_id)
        previous_model = self.registry.get_model(previous_model_id)
        
        if not current_model or not previous_model:
            logger.error("One or both models not found")
            return False
        
        # Update statuses
        self.registry.update_deployment_status(current_model_id, "retired")
        self.registry.update_deployment_status(previous_model_id, "production")
        
        logger.info(f"Rollback completed: {previous_model_id} is now in production")
        return True
    
    def compare_models(self, model_ids: List[str], X_test: pd.DataFrame,
                      y_test: pd.Series) -> pd.DataFrame:
        """Compare multiple models on test data.
        
        Args:
            model_ids: List of model IDs to compare
            X_test: Test features
            y_test: Test labels
            
        Returns:
            DataFrame with comparison results
        """
        logger.info(f"Comparing {len(model_ids)} models")
        
        models = []
        for model_id in model_ids:
            model = self.registry.get_model(model_id)
            if model:
                models.append(model)
            else:
                logger.warning(f"Model not found: {model_id}")
        
        if not models:
            raise ValueError("No valid models found for comparison")
        
        return self.evaluator.compare_models(models, X_test, y_test)
    
    def monitor_model_performance(self, model_id: str, X_recent: pd.DataFrame,
                                 y_recent: pd.Series, 
                                 performance_threshold: float = 0.05) -> Dict[str, Any]:
        """Monitor model performance and detect degradation.
        
        Args:
            model_id: Model identifier
            X_recent: Recent features
            y_recent: Recent labels
            performance_threshold: Threshold for performance degradation
            
        Returns:
            Monitoring results
        """
        logger.info(f"Monitoring performance for model: {model_id}")
        
        # Get model and metadata
        model = self.registry.get_model(model_id)
        metadata = self.registry.get_metadata(model_id)
        
        if not model or not metadata:
            raise ValueError(f"Model or metadata not found: {model_id}")
        
        # Evaluate current performance
        current_metrics = self.evaluator.evaluate_model(model, X_recent, y_recent)
        
        # Compare with validation metrics
        validation_metrics = metadata.validation_metrics
        
        # Calculate performance degradation
        degradation = {}
        alerts = []
        
        for metric in ['auc', 'precision', 'recall', 'f1']:
            if metric in validation_metrics and metric in current_metrics:
                val_score = validation_metrics[metric]
                current_score = current_metrics[metric]
                degradation[metric] = val_score - current_score
                
                if degradation[metric] > performance_threshold:
                    alerts.append(f"{metric.upper()} degraded by {degradation[metric]:.3f}")
        
        # Check for data drift (simplified)
        feature_drift = self._detect_feature_drift(X_recent, metadata.training_features)
        
        monitoring_results = {
            'model_id': model_id,
            'monitoring_date': datetime.now().isoformat(),
            'current_metrics': current_metrics,
            'validation_metrics': validation_metrics,
            'performance_degradation': degradation,
            'feature_drift': feature_drift,
            'alerts': alerts,
            'requires_retraining': len(alerts) > 0 or feature_drift['drift_detected']
        }
        
        logger.info(f"Monitoring completed. Alerts: {len(alerts)}")
        return monitoring_results
    
    def _calculate_data_hash(self, X: pd.DataFrame, y: pd.Series) -> str:
        """Calculate hash of training data for tracking."""
        import hashlib
        
        # Combine features and labels
        data_str = f"{X.shape}_{y.shape}_{X.columns.tolist()}_{y.sum()}"
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def _detect_feature_drift(self, X_current: pd.DataFrame, 
                             training_features: List[str]) -> Dict[str, Any]:
        """Detect feature drift (simplified implementation).
        
        Args:
            X_current: Current feature data
            training_features: Original training features
            
        Returns:
            Drift detection results
        """
        drift_results = {
            'drift_detected': False,
            'missing_features': [],
            'new_features': [],
            'feature_stats': {}
        }
        
        current_features = set(X_current.columns)
        training_features_set = set(training_features)
        
        # Check for missing/new features
        drift_results['missing_features'] = list(training_features_set - current_features)
        drift_results['new_features'] = list(current_features - training_features_set)
        
        # Basic statistical drift detection
        for feature in training_features:
            if feature in X_current.columns:
                feature_data = X_current[feature]
                
                # Calculate basic statistics
                stats = {
                    'mean': float(feature_data.mean()),
                    'std': float(feature_data.std()),
                    'min': float(feature_data.min()),
                    'max': float(feature_data.max()),
                    'null_rate': float(feature_data.isnull().mean())
                }
                
                drift_results['feature_stats'][feature] = stats
        
        # Set drift detected if there are missing/new features
        if drift_results['missing_features'] or drift_results['new_features']:
            drift_results['drift_detected'] = True
        
        return drift_results
    
    def get_production_model(self) -> Optional[FraudDetectionModel]:
        """Get current production model.
        
        Returns:
            Production model or None if not found
        """
        production_models = self.registry.list_models(status="production")
        
        if not production_models:
            logger.warning("No production model found")
            return None
        
        # Return the most recently deployed production model
        latest_model = max(production_models, key=lambda x: x.deployment_date or x.created_at)
        
        return self.registry.get_model(latest_model.model_id)
    
    def cleanup_old_models(self, keep_days: int = 30, 
                          keep_production: bool = True) -> int:
        """Clean up old models to save storage space.
        
        Args:
            keep_days: Number of days to keep models
            keep_production: Whether to keep production models
            
        Returns:
            Number of models deleted
        """
        logger.info(f"Cleaning up models older than {keep_days} days")
        
        cutoff_date = datetime.now() - timedelta(days=keep_days)
        models_to_delete = []
        
        for model_id, metadata in self.registry.models_metadata.items():
            # Skip if model is too recent
            if metadata.created_at > cutoff_date:
                continue
            
            # Skip production models if requested
            if keep_production and metadata.deployment_status == "production":
                continue
            
            models_to_delete.append(model_id)
        
        # Delete models
        deleted_count = 0
        for model_id in models_to_delete:
            if self.registry.delete_model(model_id):
                deleted_count += 1
        
        logger.info(f"Deleted {deleted_count} old models")
        return deleted_count
    
    def export_model_report(self, model_id: str, 
                           output_path: Optional[str] = None) -> str:
        """Export comprehensive model report.
        
        Args:
            model_id: Model identifier
            output_path: Output file path
            
        Returns:
            Report content as string
        """
        metadata = self.registry.get_metadata(model_id)
        if not metadata:
            raise ValueError(f"Model not found: {model_id}")
        
        report = f"""
{'='*80}
MODEL REPORT
{'='*80}

Model Information:
  • Model ID:           {metadata.model_id}
  • Model Name:         {metadata.model_name}
  • Model Version:      {metadata.model_version}
  • Model Type:         {metadata.model_type}
  • Created At:         {metadata.created_at.strftime('%Y-%m-%d %H:%M:%S')}
  • Created By:         {metadata.created_by}
  • Deployment Status:  {metadata.deployment_status}

Training Information:
  • Training Samples:   {metadata.training_samples:,}
  • Number of Features: {len(metadata.training_features)}
  • Data Hash:          {metadata.training_data_hash}

Hyperparameters:
"""
        
        for param, value in metadata.hyperparameters.items():
            report += f"  • {param}: {value}\n"
        
        report += "\nValidation Metrics:\n"
        for metric, value in metadata.validation_metrics.items():
            if isinstance(value, float):
                report += f"  • {metric}: {value:.4f}\n"
            else:
                report += f"  • {metric}: {value}\n"
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
            logger.info(f"Report exported to: {output_path}")
        
        return report
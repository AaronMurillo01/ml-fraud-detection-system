#!/usr/bin/env python3
"""Model training pipeline for fraud detection system.

This module provides comprehensive model training capabilities including:
- Data preprocessing and feature engineering
- Model training with hyperparameter optimization
- Model validation and evaluation
- Model versioning and artifact management
- Ensemble model creation
"""

import os
import json
import pickle
import logging
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, roc_curve, average_precision_score
)
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import xgboost as xgb
import lightgbm as lgb
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm

# Import project modules
sys.path.append(str(Path(__file__).parent.parent))
from shared.models import Transaction, ModelScore
from service.ml_service import FraudDetectionModel, ModelEnsemble
from features.feature_engineering import FeatureEngineer
from training.data_loader import DataLoader
from training.model_evaluator import ModelEvaluator


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    
    # Data configuration
    data_path: str = "data/transactions.csv"
    test_size: float = 0.2
    validation_size: float = 0.2
    random_state: int = 42
    
    # Feature engineering
    feature_selection: bool = True
    feature_importance_threshold: float = 0.001
    
    # Model configuration
    models_to_train: List[str] = None
    enable_hyperparameter_tuning: bool = True
    cv_folds: int = 5
    
    # Training configuration
    max_evals: int = 100
    early_stopping_rounds: int = 50
    
    # Output configuration
    model_output_dir: str = "models"
    experiment_name: str = "fraud_detection"
    
    def __post_init__(self):
        if self.models_to_train is None:
            self.models_to_train = [
                "xgboost",
                "lightgbm", 
                "random_forest",
                "logistic_regression"
            ]


class ModelTrainer:
    """Main model training class."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.data_loader = DataLoader()
        self.feature_engineer = FeatureEngineer()
        self.evaluator = ModelEvaluator()
        
        # Create output directory
        self.model_dir = Path(config.model_output_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize MLflow
        mlflow.set_experiment(config.experiment_name)
        
        # Model configurations
        self.model_configs = {
            "xgboost": {
                "class": xgb.XGBClassifier,
                "default_params": {
                    "objective": "binary:logistic",
                    "eval_metric": "auc",
                    "random_state": config.random_state,
                    "n_estimators": 100
                },
                "param_space": {
                    "n_estimators": hp.choice("n_estimators", [100, 200, 300, 500]),
                    "max_depth": hp.choice("max_depth", [3, 4, 5, 6, 7]),
                    "learning_rate": hp.uniform("learning_rate", 0.01, 0.3),
                    "subsample": hp.uniform("subsample", 0.6, 1.0),
                    "colsample_bytree": hp.uniform("colsample_bytree", 0.6, 1.0),
                    "reg_alpha": hp.uniform("reg_alpha", 0, 10),
                    "reg_lambda": hp.uniform("reg_lambda", 1, 10)
                }
            },
            "lightgbm": {
                "class": lgb.LGBMClassifier,
                "default_params": {
                    "objective": "binary",
                    "metric": "auc",
                    "random_state": config.random_state,
                    "n_estimators": 100,
                    "verbose": -1
                },
                "param_space": {
                    "n_estimators": hp.choice("n_estimators", [100, 200, 300, 500]),
                    "max_depth": hp.choice("max_depth", [3, 4, 5, 6, 7]),
                    "learning_rate": hp.uniform("learning_rate", 0.01, 0.3),
                    "subsample": hp.uniform("subsample", 0.6, 1.0),
                    "colsample_bytree": hp.uniform("colsample_bytree", 0.6, 1.0),
                    "reg_alpha": hp.uniform("reg_alpha", 0, 10),
                    "reg_lambda": hp.uniform("reg_lambda", 1, 10)
                }
            },
            "random_forest": {
                "class": RandomForestClassifier,
                "default_params": {
                    "random_state": config.random_state,
                    "n_estimators": 100
                },
                "param_space": {
                    "n_estimators": hp.choice("n_estimators", [100, 200, 300, 500]),
                    "max_depth": hp.choice("max_depth", [5, 10, 15, 20, None]),
                    "min_samples_split": hp.choice("min_samples_split", [2, 5, 10]),
                    "min_samples_leaf": hp.choice("min_samples_leaf", [1, 2, 4]),
                    "max_features": hp.choice("max_features", ["sqrt", "log2", None])
                }
            },
            "logistic_regression": {
                "class": LogisticRegression,
                "default_params": {
                    "random_state": config.random_state,
                    "max_iter": 1000
                },
                "param_space": {
                    "C": hp.loguniform("C", np.log(0.01), np.log(100)),
                    "penalty": hp.choice("penalty", ["l1", "l2", "elasticnet"]),
                    "solver": hp.choice("solver", ["liblinear", "saga"]),
                    "l1_ratio": hp.uniform("l1_ratio", 0, 1)
                }
            }
        }
    
    def load_and_prepare_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Load and prepare training data."""
        logger.info("Loading and preparing data...")
        
        # Load raw data
        df = self.data_loader.load_training_data(self.config.data_path)
        
        # Basic data validation
        if df.empty:
            raise ValueError("No data loaded")
        
        logger.info(f"Loaded {len(df)} transactions")
        logger.info(f"Fraud rate: {df['is_fraud'].mean():.4f}")
        
        # Feature engineering
        logger.info("Engineering features...")
        features_df = self.feature_engineer.engineer_features(df)
        
        # Prepare features and target
        target_col = 'is_fraud'
        feature_cols = [col for col in features_df.columns if col != target_col]
        
        X = features_df[feature_cols]
        y = features_df[target_col]
        
        # Handle missing values
        X = X.fillna(0)
        
        logger.info(f"Prepared {X.shape[1]} features for {X.shape[0]} samples")
        
        return X, y
    
    def split_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, ...]:
        """Split data into train, validation, and test sets."""
        logger.info("Splitting data...")
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y
        )
        
        # Second split: train vs validation
        val_size_adjusted = self.config.validation_size / (1 - self.config.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=self.config.random_state,
            stratify=y_temp
        )
        
        logger.info(f"Train set: {X_train.shape[0]} samples")
        logger.info(f"Validation set: {X_val.shape[0]} samples")
        logger.info(f"Test set: {X_test.shape[0]} samples")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def optimize_hyperparameters(self, model_name: str, X_train: pd.DataFrame, 
                                y_train: pd.Series, X_val: pd.DataFrame, 
                                y_val: pd.Series) -> Dict[str, Any]:
        """Optimize hyperparameters using Hyperopt."""
        logger.info(f"Optimizing hyperparameters for {model_name}...")
        
        model_config = self.model_configs[model_name]
        param_space = model_config["param_space"]
        
        def objective(params):
            """Objective function for hyperparameter optimization."""
            try:
                # Merge with default parameters
                full_params = {**model_config["default_params"], **params}
                
                # Handle special cases for logistic regression
                if model_name == "logistic_regression":
                    if full_params["penalty"] == "elasticnet":
                        full_params["solver"] = "saga"
                    elif full_params["penalty"] == "l1":
                        full_params["solver"] = "liblinear"
                        full_params.pop("l1_ratio", None)
                    else:
                        full_params.pop("l1_ratio", None)
                
                # Create and train model
                model = model_config["class"](**full_params)
                model.fit(X_train, y_train)
                
                # Evaluate on validation set
                y_pred_proba = model.predict_proba(X_val)[:, 1]
                auc_score = roc_auc_score(y_val, y_pred_proba)
                
                # Return negative AUC (hyperopt minimizes)
                return {"loss": -auc_score, "status": STATUS_OK}
            
            except Exception as e:
                logger.warning(f"Error in hyperparameter optimization: {e}")
                return {"loss": 0, "status": STATUS_OK}
        
        # Run optimization
        trials = Trials()
        best_params = fmin(
            fn=objective,
            space=param_space,
            algo=tpe.suggest,
            max_evals=self.config.max_evals,
            trials=trials,
            verbose=False
        )
        
        # Get best score
        best_score = -min([trial["result"]["loss"] for trial in trials.trials])
        logger.info(f"Best validation AUC for {model_name}: {best_score:.4f}")
        
        return best_params
    
    def train_single_model(self, model_name: str, X_train: pd.DataFrame, 
                          y_train: pd.Series, X_val: pd.DataFrame, 
                          y_val: pd.Series) -> FraudDetectionModel:
        """Train a single model."""
        logger.info(f"Training {model_name} model...")
        
        with mlflow.start_run(run_name=f"{model_name}_training"):
            model_config = self.model_configs[model_name]
            
            # Get parameters
            if self.config.enable_hyperparameter_tuning:
                best_params = self.optimize_hyperparameters(
                    model_name, X_train, y_train, X_val, y_val
                )
                params = {**model_config["default_params"], **best_params}
            else:
                params = model_config["default_params"]
            
            # Log parameters
            mlflow.log_params(params)
            
            # Create and train model
            sklearn_model = model_config["class"](**params)
            sklearn_model.fit(X_train, y_train)
            
            # Create FraudDetectionModel wrapper
            fraud_model = FraudDetectionModel(
                model_name=model_name,
                model_version="1.0.0"
            )
            fraud_model.model = sklearn_model
            fraud_model.feature_names = list(X_train.columns)
            fraud_model.is_trained = True
            
            # Evaluate model
            metrics = self.evaluator.evaluate_model(
                fraud_model, X_val, y_val
            )
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Log model
            if model_name in ["xgboost", "lightgbm"]:
                if model_name == "xgboost":
                    mlflow.xgboost.log_model(sklearn_model, "model")
                else:
                    mlflow.lightgbm.log_model(sklearn_model, "model")
            else:
                mlflow.sklearn.log_model(sklearn_model, "model")
            
            # Save model locally
            model_path = self.model_dir / f"{model_name}_v1.0.0.pkl"
            fraud_model.save_model(str(model_path))
            
            logger.info(f"Model {model_name} trained successfully")
            logger.info(f"Validation AUC: {metrics.get('auc', 'N/A'):.4f}")
            
            return fraud_model
    
    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                        X_val: pd.DataFrame, y_val: pd.Series) -> List[FraudDetectionModel]:
        """Train all specified models."""
        logger.info("Training all models...")
        
        trained_models = []
        
        for model_name in self.config.models_to_train:
            try:
                model = self.train_single_model(
                    model_name, X_train, y_train, X_val, y_val
                )
                trained_models.append(model)
            except Exception as e:
                logger.error(f"Failed to train {model_name}: {e}")
                continue
        
        logger.info(f"Successfully trained {len(trained_models)} models")
        return trained_models
    
    def create_ensemble(self, models: List[FraudDetectionModel],
                       X_val: pd.DataFrame, y_val: pd.Series) -> ModelEnsemble:
        """Create and optimize ensemble model."""
        logger.info("Creating ensemble model...")
        
        with mlflow.start_run(run_name="ensemble_training"):
            # Create ensemble with equal weights initially
            ensemble = ModelEnsemble(
                models=models,
                voting_strategy="soft",
                weights=None  # Equal weights
            )
            
            # Evaluate ensemble
            ensemble_metrics = self.evaluator.evaluate_ensemble(
                ensemble, X_val, y_val
            )
            
            # Log ensemble metrics
            mlflow.log_metrics(ensemble_metrics)
            
            # Save ensemble
            ensemble_path = self.model_dir / "ensemble_v1.0.0.pkl"
            with open(ensemble_path, "wb") as f:
                pickle.dump(ensemble, f)
            
            logger.info("Ensemble model created successfully")
            logger.info(f"Ensemble validation AUC: {ensemble_metrics.get('auc', 'N/A'):.4f}")
            
            return ensemble
    
    def evaluate_final_models(self, models: List[FraudDetectionModel],
                             ensemble: ModelEnsemble, X_test: pd.DataFrame, 
                             y_test: pd.Series) -> Dict[str, Dict[str, float]]:
        """Evaluate all models on test set."""
        logger.info("Evaluating models on test set...")
        
        results = {}
        
        # Evaluate individual models
        for model in models:
            metrics = self.evaluator.evaluate_model(model, X_test, y_test)
            results[model.model_name] = metrics
            logger.info(f"{model.model_name} test AUC: {metrics.get('auc', 'N/A'):.4f}")
        
        # Evaluate ensemble
        ensemble_metrics = self.evaluator.evaluate_ensemble(ensemble, X_test, y_test)
        results["ensemble"] = ensemble_metrics
        logger.info(f"Ensemble test AUC: {ensemble_metrics.get('auc', 'N/A'):.4f}")
        
        return results
    
    def save_training_metadata(self, results: Dict[str, Dict[str, float]],
                              feature_names: List[str]) -> None:
        """Save training metadata and results."""
        metadata = {
            "training_timestamp": datetime.now(timezone.utc).isoformat(),
            "config": asdict(self.config),
            "results": results,
            "feature_names": feature_names,
            "model_versions": {
                model_name: "1.0.0" for model_name in self.config.models_to_train
            }
        }
        
        metadata_path = self.model_dir / "training_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Training metadata saved to {metadata_path}")
    
    def run_training_pipeline(self) -> None:
        """Run the complete training pipeline."""
        logger.info("Starting model training pipeline...")
        
        try:
            # Load and prepare data
            X, y = self.load_and_prepare_data()
            
            # Split data
            X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y)
            
            # Train individual models
            trained_models = self.train_all_models(X_train, y_train, X_val, y_val)
            
            if not trained_models:
                raise ValueError("No models were successfully trained")
            
            # Create ensemble
            ensemble = self.create_ensemble(trained_models, X_val, y_val)
            
            # Final evaluation
            results = self.evaluate_final_models(
                trained_models, ensemble, X_test, y_test
            )
            
            # Save metadata
            self.save_training_metadata(results, list(X.columns))
            
            logger.info("Training pipeline completed successfully!")
            
            # Print summary
            print("\n" + "="*50)
            print("TRAINING RESULTS SUMMARY")
            print("="*50)
            
            for model_name, metrics in results.items():
                print(f"{model_name.upper()}:")
                print(f"  AUC: {metrics.get('auc', 'N/A'):.4f}")
                print(f"  Precision: {metrics.get('precision', 'N/A'):.4f}")
                print(f"  Recall: {metrics.get('recall', 'N/A'):.4f}")
                print(f"  F1-Score: {metrics.get('f1', 'N/A'):.4f}")
                print()
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            raise


def main():
    """Main entry point for model training."""
    parser = argparse.ArgumentParser(
        description="Train fraud detection models",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--data-path", 
        type=str, 
        default="data/transactions.csv",
        help="Path to training data"
    )
    parser.add_argument(
        "--models", 
        nargs="+",
        choices=["xgboost", "lightgbm", "random_forest", "logistic_regression"],
        default=["xgboost", "lightgbm", "random_forest"],
        help="Models to train"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="models",
        help="Output directory for trained models"
    )
    parser.add_argument(
        "--experiment-name", 
        type=str, 
        default="fraud_detection",
        help="MLflow experiment name"
    )
    parser.add_argument(
        "--no-hyperopt", 
        action="store_true",
        help="Disable hyperparameter optimization"
    )
    parser.add_argument(
        "--max-evals", 
        type=int, 
        default=100,
        help="Maximum evaluations for hyperparameter optimization"
    )
    parser.add_argument(
        "--test-size", 
        type=float, 
        default=0.2,
        help="Test set size (0.0-1.0)"
    )
    parser.add_argument(
        "--val-size", 
        type=float, 
        default=0.2,
        help="Validation set size (0.0-1.0)"
    )
    parser.add_argument(
        "--random-state", 
        type=int, 
        default=42,
        help="Random state for reproducibility"
    )
    
    args = parser.parse_args()
    
    # Create training configuration
    config = TrainingConfig(
        data_path=args.data_path,
        models_to_train=args.models,
        model_output_dir=args.output_dir,
        experiment_name=args.experiment_name,
        enable_hyperparameter_tuning=not args.no_hyperopt,
        max_evals=args.max_evals,
        test_size=args.test_size,
        validation_size=args.val_size,
        random_state=args.random_state
    )
    
    # Run training
    trainer = ModelTrainer(config)
    trainer.run_training_pipeline()


if __name__ == "__main__":
    main()
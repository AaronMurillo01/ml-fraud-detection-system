"""XGBoost model wrapper for fraud detection.

This module provides specialized functionality for XGBoost models including
SHAP explanations, feature importance analysis, and optimized inference.
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    xgb = None

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    shap = None

from .model_loader import ModelMetadata, ModelPrediction

logger = logging.getLogger(__name__)


class XGBoostModelInfo(BaseModel):
    """XGBoost model information and configuration."""
    model_type: str = "XGBoost"
    n_estimators: int
    max_depth: int
    learning_rate: float
    subsample: float
    colsample_bytree: float
    random_state: int
    objective: str = "binary:logistic"
    eval_metric: str = "auc"
    early_stopping_rounds: Optional[int] = None
    feature_names: List[str]
    num_features: int
    num_classes: int = 2
    booster_type: str = "gbtree"


class SHAPExplanation(BaseModel):
    """SHAP explanation for a prediction."""
    shap_values: Dict[str, float] = Field(..., description="SHAP values for each feature")
    base_value: float = Field(..., description="Base prediction value")
    prediction_value: float = Field(..., description="Final prediction value")
    top_positive_features: List[Tuple[str, float]] = Field(..., description="Top features increasing fraud probability")
    top_negative_features: List[Tuple[str, float]] = Field(..., description="Top features decreasing fraud probability")
    explanation_summary: str = Field(..., description="Human-readable explanation")


class XGBoostPrediction(ModelPrediction):
    """Extended prediction with XGBoost-specific information."""
    shap_explanation: Optional[SHAPExplanation] = None
    tree_prediction_path: Optional[Dict[str, Any]] = None
    leaf_indices: Optional[List[int]] = None
    model_info: Optional[XGBoostModelInfo] = None


class XGBoostModelWrapper:
    """Wrapper for XGBoost models with enhanced functionality."""
    
    def __init__(self, model_path: str, metadata: ModelMetadata):
        """Initialize XGBoost model wrapper.
        
        Args:
            model_path: Path to the XGBoost model file
            metadata: Model metadata
            
        Raises:
            ImportError: If XGBoost is not available
            FileNotFoundError: If model file doesn't exist
        """
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is not installed. Please install with: pip install xgboost")
        
        self.model_path = model_path
        self.metadata = metadata
        self.model: Optional[xgb.XGBClassifier] = None
        self.shap_explainer: Optional[Any] = None
        self.model_info: Optional[XGBoostModelInfo] = None
        
        # Load model
        self._load_model()
        
        # Initialize SHAP explainer if available
        if SHAP_AVAILABLE:
            self._initialize_shap_explainer()
        
        logger.info(f"XGBoost model wrapper initialized for {metadata.model_name}:{metadata.model_version}")
    
    def _load_model(self):
        """Load XGBoost model from file."""
        model_path = Path(self.model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        try:
            # Try loading as XGBoost native format first
            if model_path.suffix == '.json':
                self.model = xgb.XGBClassifier()
                self.model.load_model(str(model_path))
            elif model_path.suffix == '.pkl':
                # Load pickled model
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
            else:
                # Try XGBoost native format
                self.model = xgb.XGBClassifier()
                self.model.load_model(str(model_path))
            
            # Extract model information
            self._extract_model_info()
            
            logger.debug(f"XGBoost model loaded from {self.model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load XGBoost model from {self.model_path}: {e}")
            raise
    
    def _extract_model_info(self):
        """Extract model information and configuration."""
        if self.model is None:
            return
        
        try:
            # Get model parameters
            params = self.model.get_params()
            
            self.model_info = XGBoostModelInfo(
                n_estimators=params.get('n_estimators', 100),
                max_depth=params.get('max_depth', 6),
                learning_rate=params.get('learning_rate', 0.1),
                subsample=params.get('subsample', 1.0),
                colsample_bytree=params.get('colsample_bytree', 1.0),
                random_state=params.get('random_state', 0),
                objective=params.get('objective', 'binary:logistic'),
                eval_metric=params.get('eval_metric', 'auc'),
                early_stopping_rounds=params.get('early_stopping_rounds'),
                feature_names=self.metadata.feature_columns,
                num_features=len(self.metadata.feature_columns),
                booster_type=params.get('booster', 'gbtree')
            )
            
        except Exception as e:
            logger.warning(f"Failed to extract model info: {e}")
    
    def _initialize_shap_explainer(self):
        """Initialize SHAP explainer for model interpretability."""
        if not SHAP_AVAILABLE or self.model is None:
            return
        
        try:
            # Use TreeExplainer for XGBoost models
            self.shap_explainer = shap.TreeExplainer(self.model)
            logger.debug("SHAP explainer initialized")
            
        except Exception as e:
            logger.warning(f"Failed to initialize SHAP explainer: {e}")
            self.shap_explainer = None
    
    def predict(self, features: pd.DataFrame, include_shap: bool = False) -> XGBoostPrediction:
        """Make prediction with XGBoost model.
        
        Args:
            features: Input features DataFrame
            include_shap: Whether to include SHAP explanations
            
        Returns:
            XGBoost prediction with enhanced information
            
        Raises:
            ValueError: If model is not loaded or features are invalid
        """
        if self.model is None:
            raise ValueError("Model is not loaded")
        
        if features.empty:
            raise ValueError("Features DataFrame is empty")
        
        try:
            # Make prediction
            fraud_probabilities = self.model.predict_proba(features)
            fraud_score = float(fraud_probabilities[0][1])
            
            # Get feature importance
            feature_importance = {}
            if hasattr(self.model, 'feature_importances_'):
                feature_importance = dict(zip(
                    self.metadata.feature_columns,
                    self.model.feature_importances_.tolist()
                ))
            
            # Prepare model features
            model_features = {col: features.iloc[0][col] for col in self.metadata.feature_columns}
            
            # Determine risk level and decision
            risk_level, decision, decision_reason = self._classify_risk(
                fraud_score, self.metadata.threshold_config
            )
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence(fraud_score)
            
            # Get SHAP explanation if requested
            shap_explanation = None
            if include_shap and self.shap_explainer is not None:
                shap_explanation = self._get_shap_explanation(features, fraud_score)
            
            # Get tree prediction path (XGBoost specific)
            tree_prediction_path = None
            leaf_indices = None
            if hasattr(self.model, 'apply'):
                try:
                    leaf_indices = self.model.apply(features).tolist()
                    tree_prediction_path = {"leaf_indices": leaf_indices}
                except Exception as e:
                    logger.debug(f"Failed to get tree prediction path: {e}")
            
            return XGBoostPrediction(
                fraud_score=fraud_score,
                risk_level=risk_level,
                confidence_score=confidence_score,
                feature_importance=feature_importance,
                model_features=model_features,
                processing_time_ms=0.0,  # Will be set by caller
                threshold_used=self.metadata.threshold_config.get('high_risk', 0.8),
                decision=decision,
                decision_reason=decision_reason,
                shap_explanation=shap_explanation,
                tree_prediction_path=tree_prediction_path,
                leaf_indices=leaf_indices,
                model_info=self.model_info
            )
            
        except Exception as e:
            logger.error(f"XGBoost prediction failed: {e}")
            raise
    
    def _get_shap_explanation(self, features: pd.DataFrame, fraud_score: float) -> SHAPExplanation:
        """Generate SHAP explanation for the prediction.
        
        Args:
            features: Input features
            fraud_score: Predicted fraud score
            
        Returns:
            SHAP explanation
        """
        if self.shap_explainer is None:
            raise ValueError("SHAP explainer is not available")
        
        try:
            # Calculate SHAP values
            shap_values = self.shap_explainer.shap_values(features)
            
            # For binary classification, use the positive class SHAP values
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Positive class (fraud)
            
            # Get base value (expected value)
            base_value = float(self.shap_explainer.expected_value)
            if isinstance(base_value, (list, np.ndarray)):
                base_value = float(base_value[1])  # Positive class
            
            # Create feature-wise SHAP values dictionary
            feature_shap_values = dict(zip(
                self.metadata.feature_columns,
                shap_values[0].tolist()
            ))
            
            # Sort features by absolute SHAP value
            sorted_features = sorted(
                feature_shap_values.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )
            
            # Get top positive and negative features
            positive_features = [(name, value) for name, value in sorted_features if value > 0][:5]
            negative_features = [(name, value) for name, value in sorted_features if value < 0][:5]
            
            # Generate explanation summary
            explanation_summary = self._generate_explanation_summary(
                positive_features, negative_features, fraud_score
            )
            
            return SHAPExplanation(
                shap_values=feature_shap_values,
                base_value=base_value,
                prediction_value=fraud_score,
                top_positive_features=positive_features,
                top_negative_features=negative_features,
                explanation_summary=explanation_summary
            )
            
        except Exception as e:
            logger.error(f"SHAP explanation generation failed: {e}")
            raise
    
    def _generate_explanation_summary(self, 
                                    positive_features: List[Tuple[str, float]], 
                                    negative_features: List[Tuple[str, float]], 
                                    fraud_score: float) -> str:
        """Generate human-readable explanation summary.
        
        Args:
            positive_features: Features increasing fraud probability
            negative_features: Features decreasing fraud probability
            fraud_score: Predicted fraud score
            
        Returns:
            Human-readable explanation
        """
        summary_parts = []
        
        # Overall assessment
        if fraud_score > 0.8:
            summary_parts.append("HIGH FRAUD RISK detected.")
        elif fraud_score > 0.6:
            summary_parts.append("MEDIUM FRAUD RISK detected.")
        elif fraud_score > 0.3:
            summary_parts.append("LOW-MEDIUM FRAUD RISK detected.")
        else:
            summary_parts.append("LOW FRAUD RISK detected.")
        
        # Top contributing factors
        if positive_features:
            top_positive = positive_features[0]
            summary_parts.append(f"Primary risk factor: {top_positive[0]} (impact: +{top_positive[1]:.3f}).")
        
        if negative_features:
            top_negative = negative_features[0]
            summary_parts.append(f"Primary protective factor: {top_negative[0]} (impact: {top_negative[1]:.3f}).")
        
        # Additional context
        if len(positive_features) > 1:
            other_risks = ", ".join([f[0] for f in positive_features[1:3]])
            summary_parts.append(f"Additional risk factors: {other_risks}.")
        
        return " ".join(summary_parts)
    
    def _classify_risk(self, fraud_score: float, threshold_config: Dict[str, float]) -> Tuple[str, str, str]:
        """Classify risk level and decision based on fraud score.
        
        Args:
            fraud_score: Fraud probability score (0-1)
            threshold_config: Threshold configuration
            
        Returns:
            Tuple of (risk_level, decision, decision_reason)
        """
        low_threshold = threshold_config.get('low_risk', 0.3)
        medium_threshold = threshold_config.get('medium_risk', 0.6)
        high_threshold = threshold_config.get('high_risk', 0.8)
        critical_threshold = threshold_config.get('critical_risk', 0.95)
        
        if fraud_score < low_threshold:
            return "LOW", "APPROVE", f"Low fraud score ({fraud_score:.3f}) below threshold ({low_threshold})"
        elif fraud_score < medium_threshold:
            return "LOW", "APPROVE", f"Low-medium fraud score ({fraud_score:.3f}) within acceptable range"
        elif fraud_score < high_threshold:
            return "MEDIUM", "REVIEW", f"Medium fraud score ({fraud_score:.3f}) requires manual review"
        elif fraud_score < critical_threshold:
            return "HIGH", "DECLINE", f"High fraud score ({fraud_score:.3f}) exceeds risk tolerance"
        else:
            return "CRITICAL", "DECLINE", f"Critical fraud score ({fraud_score:.3f}) indicates high fraud risk"
    
    def _calculate_confidence(self, fraud_score: float) -> float:
        """Calculate confidence score based on prediction certainty.
        
        Args:
            fraud_score: Fraud probability score
            
        Returns:
            Confidence score (0-1)
        """
        # Base confidence from model's validation metrics
        base_confidence = self.metadata.validation_metrics.get('auc', 0.8)
        
        # Adjust confidence based on prediction certainty
        # More confident when prediction is closer to 0 or 1
        prediction_certainty = abs(fraud_score - 0.5) * 2
        
        # Combine base confidence with prediction certainty
        confidence = base_confidence * (0.7 + 0.3 * prediction_certainty)
        
        return min(confidence, 1.0)
    
    def get_feature_importance(self, importance_type: str = 'weight') -> Dict[str, float]:
        """Get feature importance from the model.
        
        Args:
            importance_type: Type of importance ('weight', 'gain', 'cover')
            
        Returns:
            Dictionary of feature importance scores
        """
        if self.model is None:
            return {}
        
        try:
            if hasattr(self.model, 'get_booster'):
                # XGBoost native method
                booster = self.model.get_booster()
                importance_dict = booster.get_score(importance_type=importance_type)
                
                # Map feature indices to names if needed
                if self.metadata.feature_columns:
                    mapped_importance = {}
                    for i, feature_name in enumerate(self.metadata.feature_columns):
                        feature_key = f'f{i}'
                        if feature_key in importance_dict:
                            mapped_importance[feature_name] = importance_dict[feature_key]
                        else:
                            mapped_importance[feature_name] = importance_dict.get(feature_name, 0.0)
                    return mapped_importance
                
                return importance_dict
            
            elif hasattr(self.model, 'feature_importances_'):
                # Scikit-learn style
                return dict(zip(
                    self.metadata.feature_columns,
                    self.model.feature_importances_.tolist()
                ))
            
            else:
                logger.warning("No feature importance method available")
                return {}
                
        except Exception as e:
            logger.error(f"Failed to get feature importance: {e}")
            return {}
    
    def get_model_stats(self) -> Dict[str, Any]:
        """Get model statistics and information.
        
        Returns:
            Dictionary with model statistics
        """
        stats = {
            "model_path": self.model_path,
            "model_loaded": self.model is not None,
            "shap_available": self.shap_explainer is not None,
            "feature_count": len(self.metadata.feature_columns),
            "model_info": self.model_info.dict() if self.model_info else None
        }
        
        if self.model is not None:
            try:
                stats["model_params"] = self.model.get_params()
            except Exception:
                pass
        
        return stats


def create_xgboost_wrapper(model_path: str, metadata: ModelMetadata) -> XGBoostModelWrapper:
    """Create XGBoost model wrapper.
    
    Args:
        model_path: Path to the XGBoost model file
        metadata: Model metadata
        
    Returns:
        XGBoost model wrapper instance
    """
    return XGBoostModelWrapper(model_path, metadata)


def is_xgboost_available() -> bool:
    """Check if XGBoost is available.
    
    Returns:
        True if XGBoost is available, False otherwise
    """
    return XGBOOST_AVAILABLE


def is_shap_available() -> bool:
    """Check if SHAP is available.
    
    Returns:
        True if SHAP is available, False otherwise
    """
    return SHAP_AVAILABLE
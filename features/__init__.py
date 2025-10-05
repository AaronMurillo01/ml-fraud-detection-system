"""Feature engineering package for fraud detection.

This package provides comprehensive feature engineering capabilities including:
- Real-time transaction enrichment
- Time-based feature extraction
- User behavior analysis
- Risk scoring features
- Statistical aggregations
"""

from .transaction_enricher import (
    TransactionEnricher,
    EnrichmentConfig,
    create_transaction_enricher
)
from .feature_extractor import (
    FeatureExtractor,
    FeatureConfig,
    TimeWindowConfig,
    create_feature_extractor
)
from .user_behavior import (
    UserBehaviorAnalyzer,
    BehaviorAnalysisConfig,
    create_behavior_analyzer
)
from .risk_features import (
    RiskFeatureCalculator,
    RiskFeatureConfig,
    create_risk_calculator
)
from .feature_pipeline import (
    FeaturePipeline,
    FeaturePipelineConfig,
    create_feature_pipeline
)

__all__ = [
    # Transaction enrichment
    "TransactionEnricher",
    "EnrichmentConfig",
    "create_transaction_enricher",
    
    # Feature extraction
    "FeatureExtractor",
    "FeatureConfig",
    "TimeWindowConfig",
    "create_feature_extractor",
    
    # User behavior analysis
    "UserBehaviorAnalyzer",
    "BehaviorAnalysisConfig",
    "create_behavior_analyzer",
    
    # Risk features
    "RiskFeatureCalculator",
    "RiskFeatureConfig",
    "create_risk_calculator",
    
    # Feature pipeline
    "FeaturePipeline",
    "FeaturePipelineConfig",
    "create_feature_pipeline",
]
"""Feature pipeline for fraud detection system.

This module provides a comprehensive feature pipeline that orchestrates:
- Transaction enrichment
- Feature extraction
- User behavior analysis
- Risk feature calculation
- Feature validation and preprocessing
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union, TYPE_CHECKING
from dataclasses import dataclass
from enum import Enum
import asyncio
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import numpy as np
from pydantic import BaseModel, Field, validator
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from service.models import Transaction, EnrichedTransaction
from .transaction_enricher import TransactionEnricher, get_enricher
from .feature_extractor import FeatureExtractor, get_feature_extractor
from .user_behavior import UserBehaviorAnalyzer, get_behavior_analyzer
from .risk_features import RiskFeatureCalculator, get_risk_calculator

if TYPE_CHECKING:
    from cache import MLCacheService

logger = logging.getLogger(__name__)


class FeaturePipelineMode(str, Enum):
    """Feature pipeline execution modes."""
    TRAINING = "training"  # Full feature extraction for model training
    INFERENCE = "inference"  # Real-time feature extraction for predictions
    BATCH = "batch"  # Batch processing for multiple transactions
    VALIDATION = "validation"  # Feature validation and quality checks


class FeatureValidationLevel(str, Enum):
    """Feature validation levels."""
    NONE = "none"  # No validation
    BASIC = "basic"  # Basic type and range checks
    STRICT = "strict"  # Comprehensive validation with statistical checks
    CUSTOM = "custom"  # Custom validation rules


@dataclass
class FeatureQualityMetrics:
    """Feature quality assessment metrics."""
    missing_rate: float
    outlier_rate: float
    variance: float
    skewness: float
    kurtosis: float
    correlation_with_target: Optional[float] = None
    information_value: Optional[float] = None
    stability_index: Optional[float] = None


class FeaturePipelineConfig(BaseModel):
    """Configuration for feature pipeline."""
    
    # Pipeline mode and validation
    mode: FeaturePipelineMode = FeaturePipelineMode.INFERENCE
    validation_level: FeatureValidationLevel = FeatureValidationLevel.BASIC
    
    # Feature selection
    enable_enrichment: bool = True
    enable_extraction: bool = True
    enable_user_behavior: bool = True
    enable_risk_features: bool = True
    
    # Performance settings
    enable_parallel_processing: bool = True
    max_workers: int = 4
    batch_size: int = 1000
    timeout_seconds: int = 30
    
    # Feature preprocessing
    enable_scaling: bool = True
    scaling_method: str = "robust"  # standard, robust, minmax
    enable_imputation: bool = True
    imputation_strategy: str = "median"  # mean, median, most_frequent, constant
    
    # Feature validation thresholds
    max_missing_rate: float = 0.5
    max_outlier_rate: float = 0.1
    min_variance_threshold: float = 0.01
    max_correlation_threshold: float = 0.95
    
    # Historical data settings
    lookback_days: int = 90
    min_historical_transactions: int = 10
    
    # Feature caching
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600
    
    # Output settings
    include_metadata: bool = True
    include_quality_metrics: bool = False
    feature_name_prefix: str = ""
    
    class Config:
        use_enum_values = True


class ProcessedFeatures(BaseModel):
    """Processed features output from pipeline."""
    
    # Core feature data
    features: Dict[str, Any] = Field(default_factory=dict)
    feature_names: List[str] = Field(default_factory=list)
    feature_vector: Optional[List[float]] = None
    
    # Metadata
    transaction_id: str
    user_id: str
    processing_timestamp: datetime = Field(default_factory=datetime.utcnow)
    pipeline_version: str = "1.0.0"
    
    # Processing information
    processing_time_ms: float = 0.0
    features_extracted: int = 0
    validation_passed: bool = True
    validation_errors: List[str] = Field(default_factory=list)
    
    # Quality metrics (optional)
    quality_metrics: Optional[Dict[str, FeatureQualityMetrics]] = None
    
    # Component outputs (optional)
    enriched_transaction: Optional[EnrichedTransaction] = None
    extracted_features: Optional[Dict[str, Any]] = None
    behavior_features: Optional[Dict[str, Any]] = None
    risk_features: Optional[Dict[str, Any]] = None
    
    class Config:
        use_enum_values = True
        arbitrary_types_allowed = True


class FeaturePipeline:
    """Comprehensive feature pipeline for fraud detection."""
    
    def __init__(self, config: FeaturePipelineConfig):
        """Initialize feature pipeline.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
        
        # Initialize components
        self.enricher = get_enricher() if config.enable_enrichment else None
        self.extractor = get_feature_extractor() if config.enable_extraction else None
        self.behavior_analyzer = get_behavior_analyzer() if config.enable_user_behavior else None
        self.risk_calculator = get_risk_calculator() if config.enable_risk_features else None
        
        # Initialize preprocessing components
        self.scaler = None
        self.imputer = None
        self.preprocessor = None
        
        if config.enable_scaling or config.enable_imputation:
            self._initialize_preprocessors()
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=config.max_workers) if config.enable_parallel_processing else None

        # Feature cache (legacy in-memory cache)
        self.feature_cache = {} if config.enable_caching else None

        # Redis-based ML cache service
        self._ml_cache: Optional[MLCacheService] = None

        logger.info(f"Feature pipeline initialized with mode: {config.mode}")

    async def _ensure_cache(self):
        """Ensure ML cache service is available."""
        if self._ml_cache is None:
            from cache import get_ml_cache_service
            self._ml_cache = await get_ml_cache_service()

    def _initialize_preprocessors(self):
        """Initialize preprocessing components."""
        # Initialize scaler
        if self.config.enable_scaling:
            if self.config.scaling_method == "standard":
                self.scaler = StandardScaler()
            elif self.config.scaling_method == "robust":
                self.scaler = RobustScaler()
            elif self.config.scaling_method == "minmax":
                self.scaler = MinMaxScaler()
            else:
                logger.warning(f"Unknown scaling method: {self.config.scaling_method}")
        
        # Initialize imputer
        if self.config.enable_imputation:
            self.imputer = SimpleImputer(strategy=self.config.imputation_strategy)
    
    async def process_transaction(self, 
                                transaction: Transaction,
                                historical_data: Optional[pd.DataFrame] = None,
                                user_profile: Optional[Dict[str, Any]] = None) -> ProcessedFeatures:
        """Process a single transaction through the feature pipeline.
        
        Args:
            transaction: Transaction to process
            historical_data: Historical transaction data
            user_profile: User behavior profile
            
        Returns:
            Processed features
        """
        start_time = datetime.utcnow()
        
        try:
            # Check Redis cache first
            await self._ensure_cache()
            cached_features = await self._ml_cache.get_cached_features(transaction)

            if cached_features:
                logger.debug(f"Returning cached features for transaction {transaction.transaction_id}")
                return cached_features

            # Check legacy in-memory cache as fallback
            cache_key = f"{transaction.transaction_id}_{transaction.user_id}"
            if self.feature_cache and cache_key in self.feature_cache:
                cached_result = self.feature_cache[cache_key]
                if (datetime.utcnow() - cached_result['timestamp']).seconds < self.config.cache_ttl_seconds:
                    logger.debug(f"Returning legacy cached features for transaction {transaction.transaction_id}")
                    return cached_result['features']
            
            # Initialize result
            result = ProcessedFeatures(
                transaction_id=transaction.transaction_id,
                user_id=transaction.user_id
            )
            
            # Step 1: Transaction enrichment
            enriched_transaction = transaction
            if self.enricher:
                enriched_transaction = await self._run_with_timeout(
                    self.enricher.enrich_transaction,
                    transaction, historical_data, user_profile
                )
                if self.config.include_metadata:
                    result.enriched_transaction = enriched_transaction
            
            # Step 2: Feature extraction (parallel execution for performance)
            # Create async tasks for different feature extraction components
            feature_tasks = []

            # Statistical and time-based features (velocity, aggregations, etc.)
            if self.extractor:
                feature_tasks.append(self._extract_features(enriched_transaction, historical_data))

            # User behavior analysis (spending patterns, anomaly detection)
            if self.behavior_analyzer:
                feature_tasks.append(self._analyze_behavior(enriched_transaction, historical_data, user_profile))

            # Risk-specific features (location risk, merchant risk, etc.)
            if self.risk_calculator:
                feature_tasks.append(self._calculate_risk_features(enriched_transaction, historical_data, user_profile))

            # Execute feature extraction tasks - parallel vs sequential based on config
            if feature_tasks:
                if self.config.enable_parallel_processing:
                    # Parallel execution for better performance - all tasks run concurrently
                    # return_exceptions=True ensures one failure doesn't stop others
                    feature_results = await asyncio.gather(*feature_tasks, return_exceptions=True)
                else:
                    # Sequential execution for debugging or resource constraints
                    feature_results = []
                    for task in feature_tasks:
                        try:
                            result_item = await task
                            feature_results.append(result_item)
                        except Exception as e:
                            # Capture exceptions to continue processing other features
                            feature_results.append(e)
            else:
                feature_results = []

            # Step 3: Combine features from all components
            combined_features = {}

            # Add basic transaction features (amount, timestamp, etc.)
            # These are always included as they're fundamental to fraud detection
            combined_features.update(self._get_basic_features(enriched_transaction))

            # Add component features with error handling
            for i, task_result in enumerate(feature_results):
                if isinstance(task_result, Exception):
                    # Log error but continue processing - partial features better than none
                    logger.error(f"Feature extraction task {i} failed: {task_result}")
                    result.validation_errors.append(f"Task {i} failed: {str(task_result)}")
                    continue

                if isinstance(task_result, dict):
                    # Add prefix to avoid name conflicts between components
                    component_name = ['extracted', 'behavior', 'risk'][i] if i < 3 else f'component_{i}'
                    prefixed_features = {
                        f"{component_name}_{k}" if self.config.feature_name_prefix else k: v 
                        for k, v in task_result.items()
                    }
                    combined_features.update(prefixed_features)
                    
                    # Store component results if requested
                    if self.config.include_metadata:
                        if i == 0 and self.extractor:
                            result.extracted_features = task_result
                        elif i == 1 and self.behavior_analyzer:
                            result.behavior_features = task_result
                        elif i == 2 and self.risk_calculator:
                            result.risk_features = task_result
            
            # Step 4: Feature validation
            if self.config.validation_level != FeatureValidationLevel.NONE:
                validation_result = self._validate_features(combined_features)
                result.validation_passed = validation_result['passed']
                result.validation_errors.extend(validation_result['errors'])
            
            # Step 5: Feature preprocessing
            if self.config.enable_scaling or self.config.enable_imputation:
                combined_features = self._preprocess_features(combined_features)
            
            # Step 6: Finalize result
            result.features = combined_features
            result.feature_names = list(combined_features.keys())
            result.features_extracted = len(combined_features)
            
            # Create feature vector
            result.feature_vector = [float(v) if isinstance(v, (int, float)) else 0.0 
                                   for v in combined_features.values()]
            
            # Calculate processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            result.processing_time_ms = processing_time
            
            # Calculate quality metrics if requested
            if self.config.include_quality_metrics:
                result.quality_metrics = self._calculate_quality_metrics(combined_features)
            
            # Cache result in Redis
            try:
                await self._ml_cache.cache_features(transaction, result)
                logger.debug(f"Cached features for transaction {transaction.transaction_id}")
            except Exception as cache_error:
                logger.warning(f"Failed to cache features: {cache_error}")

            # Cache result in legacy cache as fallback
            if self.feature_cache:
                self.feature_cache[cache_key] = {
                    'features': result,
                    'timestamp': datetime.utcnow()
                }

            logger.debug(f"Processed transaction {transaction.transaction_id} in {processing_time:.2f}ms")
            return result
            
        except Exception as e:
            logger.error(f"Feature pipeline processing failed for transaction {transaction.transaction_id}: {e}")
            # Return minimal result with error information
            return ProcessedFeatures(
                transaction_id=transaction.transaction_id,
                user_id=transaction.user_id,
                validation_passed=False,
                validation_errors=[f"Pipeline processing failed: {str(e)}"]
            )
    
    async def process_batch(self, 
                          transactions: List[Transaction],
                          historical_data: Optional[pd.DataFrame] = None,
                          user_profiles: Optional[Dict[str, Dict[str, Any]]] = None) -> List[ProcessedFeatures]:
        """Process multiple transactions in batch.
        
        Args:
            transactions: List of transactions to process
            historical_data: Historical transaction data
            user_profiles: Dictionary of user profiles by user_id
            
        Returns:
            List of processed features
        """
        logger.info(f"Processing batch of {len(transactions)} transactions")
        
        # Split into batches
        batches = [transactions[i:i + self.config.batch_size] 
                  for i in range(0, len(transactions), self.config.batch_size)]
        
        results = []
        
        for batch_idx, batch in enumerate(batches):
            logger.debug(f"Processing batch {batch_idx + 1}/{len(batches)}")
            
            # Process batch
            batch_tasks = []
            for transaction in batch:
                user_profile = user_profiles.get(transaction.user_id) if user_profiles else None
                task = self.process_transaction(transaction, historical_data, user_profile)
                batch_tasks.append(task)
            
            # Execute batch
            if self.config.enable_parallel_processing:
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            else:
                batch_results = []
                for task in batch_tasks:
                    try:
                        result = await task
                        batch_results.append(result)
                    except Exception as e:
                        logger.error(f"Batch processing task failed: {e}")
                        batch_results.append(e)
            
            # Collect results
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Batch processing failed: {result}")
                    continue
                results.append(result)
        
        logger.info(f"Completed batch processing: {len(results)} successful, {len(transactions) - len(results)} failed")
        return results
    
    async def _extract_features(self, 
                              transaction: EnrichedTransaction, 
                              historical_data: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """Extract features using feature extractor.
        
        Args:
            transaction: Enriched transaction
            historical_data: Historical data
            
        Returns:
            Extracted features
        """
        if not self.extractor:
            return {}
        
        return await self._run_with_timeout(
            self.extractor.extract_features,
            transaction, historical_data
        )
    
    async def _analyze_behavior(self, 
                              transaction: EnrichedTransaction, 
                              historical_data: Optional[pd.DataFrame],
                              user_profile: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze user behavior.
        
        Args:
            transaction: Enriched transaction
            historical_data: Historical data
            user_profile: User profile
            
        Returns:
            Behavior features
        """
        if not self.behavior_analyzer:
            return {}
        
        return await self._run_with_timeout(
            self.behavior_analyzer.analyze_transaction,
            transaction, historical_data, user_profile
        )
    
    async def _calculate_risk_features(self, 
                                     transaction: EnrichedTransaction, 
                                     historical_data: Optional[pd.DataFrame],
                                     user_profile: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate risk features.
        
        Args:
            transaction: Enriched transaction
            historical_data: Historical data
            user_profile: User profile
            
        Returns:
            Risk features
        """
        if not self.risk_calculator:
            return {}
        
        return await self._run_with_timeout(
            self.risk_calculator.calculate_risk_features,
            transaction, historical_data, user_profile
        )
    
    async def _run_with_timeout(self, func, *args, **kwargs):
        """Run function with timeout.
        
        Args:
            func: Function to run
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
        """
        try:
            if asyncio.iscoroutinefunction(func):
                return await asyncio.wait_for(func(*args, **kwargs), timeout=self.config.timeout_seconds)
            else:
                loop = asyncio.get_event_loop()
                return await asyncio.wait_for(
                    loop.run_in_executor(self.executor, func, *args, **kwargs),
                    timeout=self.config.timeout_seconds
                )
        except asyncio.TimeoutError:
            logger.error(f"Function {func.__name__} timed out after {self.config.timeout_seconds} seconds")
            raise
    
    def _get_basic_features(self, transaction: EnrichedTransaction) -> Dict[str, Any]:
        """Extract basic features from transaction.
        
        Args:
            transaction: Enriched transaction
            
        Returns:
            Basic features
        """
        features = {
            'amount': transaction.amount,
            'amount_log': np.log10(max(transaction.amount, 1.0)),
            'hour_of_day': transaction.timestamp.hour,
            'day_of_week': transaction.timestamp.weekday(),
            'is_weekend': transaction.timestamp.weekday() >= 5,
        }
        
        # Add enriched features if available
        if hasattr(transaction, 'merchant_category') and transaction.merchant_category:
            features['merchant_category'] = transaction.merchant_category
        
        if hasattr(transaction, 'transaction_country') and transaction.transaction_country:
            features['transaction_country'] = transaction.transaction_country
        
        return features
    
    def _validate_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Validate extracted features.
        
        Args:
            features: Features to validate
            
        Returns:
            Validation result
        """
        errors = []
        
        # Basic validation
        if self.config.validation_level in [FeatureValidationLevel.BASIC, FeatureValidationLevel.STRICT]:
            # Check for missing values
            missing_count = sum(1 for v in features.values() if v is None or (isinstance(v, float) and np.isnan(v)))
            missing_rate = missing_count / len(features) if features else 0
            
            if missing_rate > self.config.max_missing_rate:
                errors.append(f"Missing value rate {missing_rate:.2f} exceeds threshold {self.config.max_missing_rate}")
            
            # Check for infinite values
            infinite_count = sum(1 for v in features.values() 
                               if isinstance(v, float) and (np.isinf(v) or np.isneginf(v)))
            if infinite_count > 0:
                errors.append(f"Found {infinite_count} infinite values")
        
        # Strict validation
        if self.config.validation_level == FeatureValidationLevel.STRICT:
            # Check feature variance
            numeric_features = {k: v for k, v in features.items() 
                              if isinstance(v, (int, float)) and not (isinstance(v, float) and (np.isnan(v) or np.isinf(v)))}
            
            if len(numeric_features) > 1:
                feature_values = list(numeric_features.values())
                variance = np.var(feature_values)
                if variance < self.config.min_variance_threshold:
                    errors.append(f"Feature variance {variance:.6f} below threshold {self.config.min_variance_threshold}")
        
        return {
            'passed': len(errors) == 0,
            'errors': errors
        }
    
    def _preprocess_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess features (scaling, imputation).
        
        Args:
            features: Features to preprocess
            
        Returns:
            Preprocessed features
        """
        # Convert to numeric array
        numeric_features = {}
        categorical_features = {}
        
        for k, v in features.items():
            if isinstance(v, (int, float)):
                numeric_features[k] = v
            else:
                categorical_features[k] = v
        
        if not numeric_features:
            return features
        
        # Create feature matrix
        feature_names = list(numeric_features.keys())
        feature_matrix = np.array([[numeric_features[name] for name in feature_names]])
        
        # Apply imputation
        if self.config.enable_imputation and self.imputer:
            try:
                # Fit and transform (for single sample, we use a simple strategy)
                feature_matrix = np.nan_to_num(feature_matrix, nan=np.nanmedian(feature_matrix))
            except Exception as e:
                logger.warning(f"Imputation failed: {e}")
        
        # Apply scaling
        if self.config.enable_scaling and self.scaler:
            try:
                # For single sample, we can't fit the scaler, so we use a simple normalization
                # In practice, the scaler would be pre-fitted on training data
                feature_matrix = (feature_matrix - np.mean(feature_matrix)) / (np.std(feature_matrix) + 1e-8)
            except Exception as e:
                logger.warning(f"Scaling failed: {e}")
        
        # Update features
        processed_features = categorical_features.copy()
        for i, name in enumerate(feature_names):
            processed_features[name] = float(feature_matrix[0, i])
        
        return processed_features
    
    def _calculate_quality_metrics(self, features: Dict[str, Any]) -> Dict[str, FeatureQualityMetrics]:
        """Calculate feature quality metrics.
        
        Args:
            features: Features to analyze
            
        Returns:
            Quality metrics for each feature
        """
        metrics = {}
        
        for name, value in features.items():
            if isinstance(value, (int, float)) and not (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
                # For single value, we can only calculate basic metrics
                metrics[name] = FeatureQualityMetrics(
                    missing_rate=0.0 if value is not None else 1.0,
                    outlier_rate=0.0,  # Can't determine from single value
                    variance=0.0,  # Can't calculate from single value
                    skewness=0.0,  # Can't calculate from single value
                    kurtosis=0.0   # Can't calculate from single value
                )
        
        return metrics
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores.
        
        Returns:
            Dictionary of feature importance scores
        """
        # This would typically come from a trained model
        # For now, return placeholder values
        return {
            'amount': 0.15,
            'velocity_risk_score': 0.12,
            'amount_risk_score': 0.10,
            'location_risk_score': 0.08,
            'merchant_risk_score': 0.07,
            'temporal_risk_score': 0.06,
            'network_risk_score': 0.05
        }
    
    def cleanup(self):
        """Cleanup pipeline resources."""
        if self.executor:
            self.executor.shutdown(wait=True)
        
        if self.feature_cache:
            self.feature_cache.clear()
        
        logger.info("Feature pipeline cleaned up")


def create_feature_pipeline(config: Optional[FeaturePipelineConfig] = None) -> FeaturePipeline:
    """Create feature pipeline with default or custom configuration.
    
    Args:
        config: Optional pipeline configuration
        
    Returns:
        Feature pipeline instance
    """
    if config is None:
        config = FeaturePipelineConfig()
    
    return FeaturePipeline(config)


# Global pipeline instance
_pipeline_instance: Optional[FeaturePipeline] = None


def get_feature_pipeline() -> FeaturePipeline:
    """Get global feature pipeline instance.
    
    Returns:
        Global pipeline instance
    """
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = create_feature_pipeline()
    return _pipeline_instance


def set_feature_pipeline(pipeline: FeaturePipeline):
    """Set global feature pipeline instance.
    
    Args:
        pipeline: Pipeline instance to set as global
    """
    global _pipeline_instance
    _pipeline_instance = pipeline
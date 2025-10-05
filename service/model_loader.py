"""Model loading and management for fraud detection.

This module handles loading, caching, and managing ML models for real-time
fraud detection inference.
"""

import logging
import pickle
import threading
import asyncio
import weakref
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, validator

from shared.models import ModelMetadata
from cache import get_ml_cache_service

logger = logging.getLogger(__name__)

# Custom exceptions
class ModelLoadError(Exception):
    """Exception raised when model loading fails."""
    pass

class ModelValidationError(Exception):
    """Exception raised when model validation fails."""
    pass


class ModelMetadata(BaseModel):
    """Model metadata information."""
    model_name: str
    model_version: str
    model_type: str
    model_path: str
    feature_columns: List[str]
    target_column: str
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    threshold_config: Dict[str, float] = Field(default_factory=dict)
    training_metrics: Dict[str, float] = Field(default_factory=dict)
    validation_metrics: Dict[str, float] = Field(default_factory=dict)
    test_metrics: Dict[str, float] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    deployed_at: Optional[datetime] = None
    model_status: str = "ACTIVE"
    notes: Optional[str] = None
    
    @validator('model_name')
    def validate_model_name(cls, v):
        if not v or not v.strip():
            raise ValueError('Model name cannot be empty')
        return v
    
    @validator('feature_columns')
    def validate_feature_columns(cls, v):
        if not v:
            raise ValueError('Feature columns cannot be empty')
        return v


class ModelPrediction(BaseModel):
    """Model prediction result."""
    fraud_score: float = Field(..., description="Fraud probability score")
    risk_level: str = Field(..., description="Risk level classification")
    confidence_score: float = Field(..., description="Model confidence")
    decision: str = Field(..., description="Final decision (APPROVE/REVIEW/DECLINE)")
    reason: str = Field(default="", description="Reason for the decision")
    feature_importance: Dict[str, float] = Field(default_factory=dict, description="Feature importance scores")
    model_features: Dict[str, Any] = Field(default_factory=dict, description="Features used for prediction")
    processing_time_ms: float = Field(default=0.0, description="Processing time in milliseconds")
    threshold_used: Optional[float] = Field(default=None, description="Threshold used for classification")
    
    @validator('fraud_score')
    def validate_fraud_score(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('Fraud score must be between 0 and 1')
        return v
    
    @validator('confidence_score')
    def validate_confidence_score(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('Confidence score must be between 0 and 1')
        return v


@dataclass
class LoadedModel:
    """Container for a loaded ML model and its metadata."""
    model: Any
    metadata: ModelMetadata
    loaded_at: datetime = None
    last_used: datetime = None
    usage_count: int = 0
    
    def __post_init__(self):
        if self.loaded_at is None:
            self.loaded_at = datetime.utcnow()
        if self.last_used is None:
            self.last_used = datetime.utcnow()


class ModelLoader:
    """Handles loading and caching of ML models with async support and performance optimizations."""

    def __init__(self,
                 model_cache_size: int = 100,
                 cache_ttl_hours: int = 24,
                 model_base_path: str = "./models",
                 max_workers: int = 4,
                 enable_redis_cache: bool = True,
                 preload_models: List[str] = None):
        """Initialize model loader.

        Args:
            model_cache_size: Maximum number of models to keep in cache
            cache_ttl_hours: Time-to-live for cached models in hours
            model_base_path: Base path for model files
            max_workers: Maximum number of worker threads for model loading
            enable_redis_cache: Whether to use Redis for model metadata caching
            preload_models: List of model names to preload on startup
        """
        self.model_cache_size = model_cache_size
        self.cache_ttl_hours = cache_ttl_hours
        self.model_base_path = model_base_path
        self.max_workers = max_workers
        self.enable_redis_cache = enable_redis_cache
        self.preload_models = preload_models or []

        # In-memory cache with weak references for automatic cleanup
        self._model_cache: Dict[str, LoadedModel] = {}
        self._cache_stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "total_requests": 0,
            "load_times": [],
            "redis_hits": 0,
            "redis_misses": 0
        }
        self._lock = asyncio.Lock()
        self._thread_lock = threading.RLock()

        # Thread pool for CPU-intensive model loading
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

        # Redis cache service
        self._ml_cache = None

        # Model loading semaphore to prevent too many concurrent loads
        self._loading_semaphore = asyncio.Semaphore(max_workers)

        # Currently loading models (to prevent duplicate loads)
        self._loading_models: Dict[str, asyncio.Future] = {}

        logger.info(f"ModelLoader initialized with cache_size={model_cache_size}, ttl={cache_ttl_hours}h, workers={max_workers}")

    async def _ensure_cache(self):
        """Ensure ML cache service is available."""
        if self.enable_redis_cache and self._ml_cache is None:
            self._ml_cache = await get_ml_cache_service()

    async def initialize(self):
        """Initialize the model loader and preload models if specified."""
        await self._ensure_cache()

        # Preload specified models
        if self.preload_models:
            logger.info(f"Preloading {len(self.preload_models)} models...")
            preload_tasks = []

            for model_name in self.preload_models:
                # Create dummy metadata for preloading (would normally come from database)
                metadata = ModelMetadata(
                    model_name=model_name,
                    model_version="latest",
                    model_type="xgboost",
                    model_path=f"{self.model_base_path}/{model_name}.pkl",
                    feature_columns=[],
                    target_column="is_fraud"
                )
                preload_tasks.append(self.load_model_async(metadata))

            try:
                await asyncio.gather(*preload_tasks, return_exceptions=True)
                logger.info("Model preloading completed")
            except Exception as e:
                logger.warning(f"Some models failed to preload: {e}")

    async def load_model_async(self, metadata: ModelMetadata) -> Any:
        """Asynchronously load a model from disk or cache.

        Args:
            metadata: Model metadata containing path and configuration

        Returns:
            Loaded model object

        Raises:
            ModelLoadError: If model loading fails
        """
        model_key = self._get_cache_key(metadata)

        async with self._lock:
            self._cache_stats["total_requests"] += 1

            # Check if model is already being loaded
            if model_key in self._loading_models:
                logger.debug(f"Model {model_key} is already being loaded, waiting...")
                return await self._loading_models[model_key]

            # Check if model is in memory cache and still valid
            if model_key in self._model_cache:
                cached_model = self._model_cache[model_key]
                if self._is_cache_valid(cached_model):
                    cached_model.last_used = datetime.utcnow()
                    self._cache_stats["cache_hits"] += 1
                    logger.debug(f"Model {model_key} loaded from memory cache")
                    return cached_model.model
                else:
                    # Remove expired model
                    del self._model_cache[model_key]

            # Check Redis cache for model metadata
            if self._ml_cache:
                cached_metadata = await self._ml_cache.get_cached_model_metadata(
                    metadata.model_name, metadata.model_version
                )
                if cached_metadata:
                    self._cache_stats["redis_hits"] += 1
                    logger.debug(f"Model metadata for {model_key} found in Redis cache")
                else:
                    self._cache_stats["redis_misses"] += 1

        # Create loading future to prevent duplicate loads
        loading_future = asyncio.create_task(self._load_model_from_disk(metadata))
        self._loading_models[model_key] = loading_future

        try:
            model = await loading_future
            return model
        finally:
            # Remove from loading models
            if model_key in self._loading_models:
                del self._loading_models[model_key]

    async def _load_model_from_disk(self, metadata: ModelMetadata) -> Any:
        """Load model from disk with performance optimizations.

        Args:
            metadata: Model metadata

        Returns:
            Loaded model object
        """
        async with self._loading_semaphore:
            start_time = datetime.utcnow()
            model_key = self._get_cache_key(metadata)

            try:
                # Load model in thread pool to avoid blocking event loop
                model = await asyncio.get_event_loop().run_in_executor(
                    self._executor,
                    self._load_model_sync,
                    metadata
                )

                # Cache the loaded model
                loaded_model = LoadedModel(
                    model=model,
                    metadata=metadata,
                    loaded_at=datetime.utcnow(),
                    last_used=datetime.utcnow()
                )

                # Add to memory cache
                async with self._lock:
                    self._model_cache[model_key] = loaded_model
                    self._cache_stats["cache_misses"] += 1

                    # Track loading time
                    load_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                    self._cache_stats["load_times"].append(load_time)

                    # Keep only recent load times for statistics
                    if len(self._cache_stats["load_times"]) > 100:
                        self._cache_stats["load_times"] = self._cache_stats["load_times"][-100:]

                    # Cleanup old models if cache is full
                    await self._cleanup_cache()

                # Cache metadata in Redis
                if self._ml_cache:
                    try:
                        metadata_dict = {
                            "model_name": metadata.model_name,
                            "model_version": metadata.model_version,
                            "model_type": metadata.model_type,
                            "feature_columns": metadata.feature_columns,
                            "target_column": metadata.target_column,
                            "hyperparameters": metadata.hyperparameters,
                            "threshold_config": metadata.threshold_config,
                            "loaded_at": start_time.isoformat(),
                            "load_time_ms": load_time
                        }
                        await self._ml_cache.cache_model_metadata(
                            metadata.model_name,
                            metadata.model_version,
                            metadata_dict
                        )
                    except Exception as cache_error:
                        logger.warning(f"Failed to cache model metadata: {cache_error}")

                logger.info(f"Model {model_key} loaded successfully in {load_time:.2f}ms")
                return model

            except Exception as e:
                logger.error(f"Failed to load model {model_key}: {e}")
                raise ModelLoadError(f"Failed to load model {metadata.model_name}: {str(e)}")

    def _load_model_sync(self, metadata: ModelMetadata) -> Any:
        """Synchronously load model from disk (runs in thread pool).

        Args:
            metadata: Model metadata

        Returns:
            Loaded model object
        """
        model_path = Path(metadata.model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        try:
            # Load model based on file extension
            if model_path.suffix == '.pkl':
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
            elif model_path.suffix == '.joblib':
                import joblib
                model = joblib.load(model_path)
            else:
                raise ModelLoadError(f"Unsupported model file format: {model_path.suffix}")

            # Validate model
            self._validate_model(model, metadata)

            return model

        except Exception as e:
            raise ModelLoadError(f"Failed to load model from {model_path}: {str(e)}")

    def load_model(self, metadata: ModelMetadata) -> Any:
        """Synchronous wrapper for async model loading (for backward compatibility).

        Args:
            metadata: Model metadata

        Returns:
            Loaded model object
        """
        # Check if we're in an async context
        try:
            loop = asyncio.get_running_loop()
            # We're in an async context, but this is a sync call
            # Create a new thread to run the async code
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, self.load_model_async(metadata))
                return future.result()
        except RuntimeError:
            # No event loop running, we can run async code directly
            return asyncio.run(self.load_model_async(metadata))

    async def _cleanup_cache(self):
        """Clean up old models from cache when it's full."""
        if len(self._model_cache) <= self.model_cache_size:
            return

        # Sort models by last used time
        models_by_usage = sorted(
            self._model_cache.items(),
            key=lambda x: x[1].last_used
        )

        # Remove oldest models
        models_to_remove = len(self._model_cache) - self.model_cache_size + 1
        for i in range(models_to_remove):
            model_key, _ = models_by_usage[i]
            del self._model_cache[model_key]
            logger.debug(f"Removed model {model_key} from cache (LRU cleanup)")

    def _validate_model(self, model: Any, metadata: ModelMetadata):
        """Validate that the loaded model is compatible with metadata.

        Args:
            model: Loaded model object
            metadata: Model metadata

        Raises:
            ModelValidationError: If model validation fails
        """
        try:
            # Check if model has required methods
            if not hasattr(model, 'predict'):
                raise ModelValidationError("Model does not have 'predict' method")

            # For XGBoost models, check feature names if available
            if hasattr(model, 'feature_names_in_') and metadata.feature_columns:
                model_features = set(model.feature_names_in_)
                expected_features = set(metadata.feature_columns)

                if model_features != expected_features:
                    missing_features = expected_features - model_features
                    extra_features = model_features - expected_features

                    error_msg = "Feature mismatch between model and metadata"
                    if missing_features:
                        error_msg += f". Missing: {missing_features}"
                    if extra_features:
                        error_msg += f". Extra: {extra_features}"

                    raise ModelValidationError(error_msg)

            logger.debug(f"Model validation passed for {metadata.model_name}")

        except Exception as e:
            raise ModelValidationError(f"Model validation failed: {str(e)}")

    async def batch_predict(self, models_and_data: List[tuple]) -> List[Any]:
        """Perform batch predictions across multiple models.

        Args:
            models_and_data: List of (metadata, input_data) tuples

        Returns:
            List of prediction results
        """
        # Load all models concurrently
        load_tasks = []
        for metadata, _ in models_and_data:
            load_tasks.append(self.load_model_async(metadata))

        models = await asyncio.gather(*load_tasks)

        # Perform predictions in thread pool
        predict_tasks = []
        for i, (_, input_data) in enumerate(models_and_data):
            task = asyncio.get_event_loop().run_in_executor(
                self._executor,
                models[i].predict,
                input_data
            )
            predict_tasks.append(task)

        predictions = await asyncio.gather(*predict_tasks)
        return predictions

    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics.

        Returns:
            Cache statistics dictionary
        """
        async with self._lock:
            total_requests = self._cache_stats["total_requests"]
            cache_hits = self._cache_stats["cache_hits"]
            cache_misses = self._cache_stats["cache_misses"]
            redis_hits = self._cache_stats["redis_hits"]
            redis_misses = self._cache_stats["redis_misses"]
            load_times = self._cache_stats["load_times"]

            hit_rate = (cache_hits / total_requests * 100) if total_requests > 0 else 0
            redis_hit_rate = (redis_hits / (redis_hits + redis_misses) * 100) if (redis_hits + redis_misses) > 0 else 0

            avg_load_time = sum(load_times) / len(load_times) if load_times else 0
            max_load_time = max(load_times) if load_times else 0
            min_load_time = min(load_times) if load_times else 0

            return {
                "cache_size": len(self._model_cache),
                "max_cache_size": self.model_cache_size,
                "total_requests": total_requests,
                "cache_hits": cache_hits,
                "cache_misses": cache_misses,
                "hit_rate_percent": round(hit_rate, 2),
                "redis_hits": redis_hits,
                "redis_misses": redis_misses,
                "redis_hit_rate_percent": round(redis_hit_rate, 2),
                "avg_load_time_ms": round(avg_load_time, 2),
                "max_load_time_ms": round(max_load_time, 2),
                "min_load_time_ms": round(min_load_time, 2),
                "models_in_cache": list(self._model_cache.keys())
            }

    async def preload_model(self, metadata: ModelMetadata) -> bool:
        """Preload a model into cache.

        Args:
            metadata: Model metadata

        Returns:
            True if preloaded successfully, False otherwise
        """
        try:
            await self.load_model_async(metadata)
            logger.info(f"Successfully preloaded model {metadata.model_name}:{metadata.model_version}")
            return True
        except Exception as e:
            logger.error(f"Failed to preload model {metadata.model_name}:{metadata.model_version}: {e}")
            return False

    async def clear_cache(self):
        """Clear all cached models."""
        async with self._lock:
            self._model_cache.clear()
            logger.info("Model cache cleared")

    async def remove_model_from_cache(self, model_name: str, model_version: str) -> bool:
        """Remove a specific model from cache.

        Args:
            model_name: Model name
            model_version: Model version

        Returns:
            True if model was removed, False if not found
        """
        model_key = f"{model_name}:{model_version}"

        async with self._lock:
            if model_key in self._model_cache:
                del self._model_cache[model_key]
                logger.info(f"Removed model {model_key} from cache")
                return True
            return False

    async def shutdown(self):
        """Shutdown the model loader and cleanup resources."""
        # Shutdown thread pool
        self._executor.shutdown(wait=True)

        # Clear cache
        await self.clear_cache()

        logger.info("ModelLoader shutdown completed")

    def _get_cache_key(self, metadata: ModelMetadata) -> str:
        """Generate cache key for model.

        Args:
            metadata: Model metadata

        Returns:
            Cache key string
        """
        return f"{metadata.model_name}:{metadata.model_version}"

    def _is_cache_valid(self, loaded_model: LoadedModel) -> bool:
        """Check if cached model is still valid.

        Args:
            loaded_model: Loaded model object

        Returns:
            True if cache is valid, False otherwise
        """
        if not loaded_model.loaded_at:
            return False

        ttl_delta = timedelta(hours=self.cache_ttl_hours)
        return datetime.utcnow() - loaded_model.loaded_at < ttl_delta


# Global model loader instance
_model_loader: Optional[ModelLoader] = None


async def get_model_loader() -> ModelLoader:
    """Get the global model loader instance.

    Returns:
        ModelLoader instance
    """
    global _model_loader

    if _model_loader is None:
        _model_loader = ModelLoader(
            model_cache_size=50,
            cache_ttl_hours=24,
            max_workers=4,
            enable_redis_cache=True,
            preload_models=["fraud_detector_v1"]  # Add models to preload
        )
        await _model_loader.initialize()

    return _model_loader


async def initialize_model_loader():
    """Initialize the global model loader."""
    await get_model_loader()
    logger.info("Model loader initialized")


async def shutdown_model_loader():
    """Shutdown the global model loader."""
    global _model_loader

    if _model_loader:
        await _model_loader.shutdown()
        _model_loader = None
        logger.info("Model loader shutdown")


class ModelPreprocessor:
    """Handles feature preprocessing for model inference."""
    
    def __init__(self, feature_columns: List[str]):
        """Initialize preprocessor.
        
        Args:
            feature_columns: List of expected feature column names
        """
        self.feature_columns = feature_columns
        logger.debug(f"ModelPreprocessor initialized with {len(feature_columns)} features")
    
    def prepare_features(self, transaction_data: Dict[str, Any]) -> pd.DataFrame:
        """Prepare features for model inference.
        
        Args:
            transaction_data: Raw transaction data
            
        Returns:
            DataFrame with prepared features
            
        Raises:
            ValueError: If required features are missing
        """
        try:
            # Create DataFrame with single row
            df = pd.DataFrame([transaction_data])
            
            # Ensure all required columns are present
            missing_columns = set(self.feature_columns) - set(df.columns)
            if missing_columns:
                raise ValueError(f"Missing required features: {missing_columns}")
            
            # Select and reorder columns to match model expectations
            df = df[self.feature_columns]
            
            # Handle missing values
            df = self._handle_missing_values(df)
            
            # Apply feature transformations
            df = self._apply_transformations(df)
            
            logger.debug(f"Prepared features for inference: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Feature preparation failed: {e}")
            raise
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in features.
        
        Args:
            df: DataFrame with features
            
        Returns:
            DataFrame with missing values handled
        """
        # Fill numeric columns with 0
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(0)
        
        # Fill categorical columns with 'unknown'
        categorical_columns = df.select_dtypes(include=['object']).columns
        df[categorical_columns] = df[categorical_columns].fillna('unknown')
        
        return df
    
    def _apply_transformations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply feature transformations.
        
        Args:
            df: DataFrame with features
            
        Returns:
            DataFrame with transformed features
        """
        # Apply categorical encoding for common categories
        categorical_mappings = {
            'merchant_category': {
                'grocery': 1, 'gas_station': 2, 'restaurant': 3, 'retail': 4,
                'online': 5, 'atm': 6, 'other': 0, 'unknown': 0
            },
            'payment_method': {
                'credit_card': 1, 'debit_card': 2, 'bank_transfer': 3,
                'digital_wallet': 4, 'other': 0, 'unknown': 0
            },
            'card_type': {
                'visa': 1, 'mastercard': 2, 'amex': 3, 'discover': 4,
                'other': 0, 'unknown': 0
            }
        }
        
        for column, mapping in categorical_mappings.items():
            if column in df.columns:
                df[column] = df[column].map(mapping).fillna(0)
        
        return df


# Global model loader instance
_model_loader: Optional[ModelLoader] = None
_loader_lock = threading.Lock()


def get_model_loader() -> ModelLoader:
    """Get global model loader instance.
    
    Returns:
        ModelLoader instance
    """
    global _model_loader
    
    if _model_loader is None:
        with _loader_lock:
            if _model_loader is None:
                _model_loader = ModelLoader()
    
    return _model_loader


def initialize_model_loader(model_cache_size: int = 100, cache_ttl_hours: int = 24, model_base_path: str = "./models") -> ModelLoader:
    """Initialize the global model loader instance.
    
    Args:
        model_cache_size: Maximum number of models to keep in cache
        cache_ttl_hours: Time-to-live for cached models in hours
        model_base_path: Base path for model files
        
    Returns:
        ModelLoader: The initialized model loader instance
    """
    global _model_loader
    _model_loader = ModelLoader(model_cache_size, cache_ttl_hours, model_base_path)
    return _model_loader
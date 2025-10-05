"""ML-specific caching service for predictions and features."""

import logging
import hashlib
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import json

from .redis_manager import get_redis_manager, RedisManager
from shared.models import Transaction
from service.models.prediction import PredictionResult
from features.feature_pipeline import ProcessedFeatures

logger = logging.getLogger(__name__)


class MLCacheService:
    """Caching service for ML predictions and features."""
    
    def __init__(self, redis_manager: RedisManager = None):
        """Initialize ML cache service.
        
        Args:
            redis_manager: Redis manager instance
        """
        self.redis_manager = redis_manager
        
        # Cache TTL settings (in seconds)
        self.prediction_ttl = 1800  # 30 minutes
        self.feature_ttl = 7200  # 2 hours
        self.model_metadata_ttl = 86400  # 24 hours
        self.user_profile_ttl = 3600  # 1 hour
        self.merchant_profile_ttl = 14400  # 4 hours
        
        # Cache key prefixes
        self.prefixes = {
            'prediction': 'fraud:pred:',
            'features': 'fraud:feat:',
            'model_meta': 'fraud:model:',
            'user_profile': 'fraud:user:',
            'merchant_profile': 'fraud:merchant:',
            'batch_prediction': 'fraud:batch:',
            'feature_importance': 'fraud:importance:'
        }
    
    async def _ensure_redis(self):
        """Ensure Redis manager is available."""
        if not self.redis_manager:
            self.redis_manager = await get_redis_manager()
    
    def _generate_transaction_hash(self, transaction: Transaction) -> str:
        """Generate a hash for a transaction for caching.
        
        Args:
            transaction: Transaction object
            
        Returns:
            Transaction hash string
        """
        # Create a deterministic hash based on transaction content
        transaction_data = {
            'user_id': transaction.user_id,
            'merchant_id': transaction.merchant_id,
            'amount': float(transaction.amount),
            'currency': transaction.currency,
            'transaction_type': transaction.transaction_type,
            'payment_method': transaction.payment_method,
            'location': transaction.location
        }
        
        # Sort keys for consistent hashing
        sorted_data = json.dumps(transaction_data, sort_keys=True)
        return hashlib.sha256(sorted_data.encode()).hexdigest()[:16]
    
    def _generate_feature_key(self, transaction: Transaction, feature_version: str = "v1") -> str:
        """Generate cache key for transaction features.
        
        Args:
            transaction: Transaction object
            feature_version: Feature extraction version
            
        Returns:
            Cache key for features
        """
        tx_hash = self._generate_transaction_hash(transaction)
        return f"{self.prefixes['features']}{feature_version}:{tx_hash}"
    
    def _generate_prediction_key(
        self, 
        transaction: Transaction, 
        model_name: str, 
        model_version: str,
        feature_version: str = "v1"
    ) -> str:
        """Generate cache key for prediction.
        
        Args:
            transaction: Transaction object
            model_name: ML model name
            model_version: ML model version
            feature_version: Feature extraction version
            
        Returns:
            Cache key for prediction
        """
        tx_hash = self._generate_transaction_hash(transaction)
        return f"{self.prefixes['prediction']}{model_name}:{model_version}:{feature_version}:{tx_hash}"
    
    async def cache_features(
        self, 
        transaction: Transaction, 
        features: ProcessedFeatures,
        feature_version: str = "v1"
    ) -> bool:
        """Cache processed features for a transaction.
        
        Args:
            transaction: Transaction object
            features: Processed features
            feature_version: Feature extraction version
            
        Returns:
            True if cached successfully, False otherwise
        """
        try:
            await self._ensure_redis()
            
            cache_key = self._generate_feature_key(transaction, feature_version)
            
            # Prepare features data for caching
            features_data = {
                'transaction_id': features.transaction_id,
                'user_id': features.user_id,
                'features': features.features,
                'feature_names': features.feature_names,
                'processing_time_ms': features.processing_time_ms,
                'cached_at': datetime.utcnow().isoformat(),
                'feature_version': feature_version
            }
            
            success = await self.redis_manager.set_json(
                cache_key, 
                features_data, 
                self.feature_ttl
            )
            
            if success:
                logger.debug(f"Cached features for transaction {transaction.transaction_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to cache features for transaction {transaction.transaction_id}: {e}")
            return False
    
    async def get_cached_features(
        self, 
        transaction: Transaction,
        feature_version: str = "v1"
    ) -> Optional[ProcessedFeatures]:
        """Get cached features for a transaction.
        
        Args:
            transaction: Transaction object
            feature_version: Feature extraction version
            
        Returns:
            Cached features or None if not found
        """
        try:
            await self._ensure_redis()
            
            cache_key = self._generate_feature_key(transaction, feature_version)
            features_data = await self.redis_manager.get_json(cache_key)
            
            if features_data:
                # Reconstruct ProcessedFeatures object
                features = ProcessedFeatures(
                    transaction_id=features_data['transaction_id'],
                    user_id=features_data['user_id'],
                    features=features_data['features'],
                    feature_names=features_data['feature_names'],
                    processing_time_ms=features_data['processing_time_ms']
                )
                
                logger.debug(f"Retrieved cached features for transaction {transaction.transaction_id}")
                return features
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get cached features for transaction {transaction.transaction_id}: {e}")
            return None
    
    async def cache_prediction(
        self,
        transaction: Transaction,
        prediction: PredictionResult,
        model_name: str,
        model_version: str,
        feature_version: str = "v1"
    ) -> bool:
        """Cache ML prediction for a transaction.
        
        Args:
            transaction: Transaction object
            prediction: Prediction result
            model_name: ML model name
            model_version: ML model version
            feature_version: Feature extraction version
            
        Returns:
            True if cached successfully, False otherwise
        """
        try:
            await self._ensure_redis()
            
            cache_key = self._generate_prediction_key(
                transaction, model_name, model_version, feature_version
            )
            
            # Prepare prediction data for caching
            prediction_data = {
                'transaction_id': prediction.transaction_id,
                'fraud_score': prediction.fraud_score,
                'risk_level': prediction.risk_level,
                'confidence_score': prediction.confidence_score,
                'decision': prediction.decision,
                'decision_reason': prediction.decision_reason,
                'feature_importance': prediction.feature_importance,
                'model_features': prediction.model_features,
                'processing_time_ms': prediction.processing_time_ms,
                'threshold_used': prediction.threshold_used,
                'model_name': model_name,
                'model_version': model_version,
                'feature_version': feature_version,
                'cached_at': datetime.utcnow().isoformat()
            }
            
            success = await self.redis_manager.set_json(
                cache_key,
                prediction_data,
                self.prediction_ttl
            )
            
            if success:
                logger.debug(f"Cached prediction for transaction {transaction.transaction_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to cache prediction for transaction {transaction.transaction_id}: {e}")
            return False
    
    async def get_cached_prediction(
        self,
        transaction: Transaction,
        model_name: str,
        model_version: str,
        feature_version: str = "v1"
    ) -> Optional[PredictionResult]:
        """Get cached prediction for a transaction.
        
        Args:
            transaction: Transaction object
            model_name: ML model name
            model_version: ML model version
            feature_version: Feature extraction version
            
        Returns:
            Cached prediction or None if not found
        """
        try:
            await self._ensure_redis()
            
            cache_key = self._generate_prediction_key(
                transaction, model_name, model_version, feature_version
            )
            
            prediction_data = await self.redis_manager.get_json(cache_key)
            
            if prediction_data:
                # Reconstruct PredictionResult object
                prediction = PredictionResult(
                    transaction_id=prediction_data['transaction_id'],
                    fraud_score=prediction_data['fraud_score'],
                    risk_level=prediction_data['risk_level'],
                    confidence_score=prediction_data['confidence_score'],
                    decision=prediction_data['decision'],
                    decision_reason=prediction_data['decision_reason'],
                    feature_importance=prediction_data['feature_importance'],
                    model_features=prediction_data['model_features'],
                    processing_time_ms=prediction_data['processing_time_ms'],
                    threshold_used=prediction_data['threshold_used']
                )
                
                logger.debug(f"Retrieved cached prediction for transaction {transaction.transaction_id}")
                return prediction
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get cached prediction for transaction {transaction.transaction_id}: {e}")
            return None
    
    async def cache_model_metadata(
        self,
        model_name: str,
        model_version: str,
        metadata: Dict[str, Any]
    ) -> bool:
        """Cache model metadata.
        
        Args:
            model_name: Model name
            model_version: Model version
            metadata: Model metadata
            
        Returns:
            True if cached successfully, False otherwise
        """
        try:
            await self._ensure_redis()
            
            cache_key = f"{self.prefixes['model_meta']}{model_name}:{model_version}"
            
            metadata_with_timestamp = {
                **metadata,
                'cached_at': datetime.utcnow().isoformat()
            }
            
            success = await self.redis_manager.set_json(
                cache_key,
                metadata_with_timestamp,
                self.model_metadata_ttl
            )
            
            if success:
                logger.debug(f"Cached metadata for model {model_name}:{model_version}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to cache model metadata for {model_name}:{model_version}: {e}")
            return False
    
    async def get_cached_model_metadata(
        self,
        model_name: str,
        model_version: str
    ) -> Optional[Dict[str, Any]]:
        """Get cached model metadata.
        
        Args:
            model_name: Model name
            model_version: Model version
            
        Returns:
            Cached metadata or None if not found
        """
        try:
            await self._ensure_redis()
            
            cache_key = f"{self.prefixes['model_meta']}{model_name}:{model_version}"
            metadata = await self.redis_manager.get_json(cache_key)
            
            if metadata:
                logger.debug(f"Retrieved cached metadata for model {model_name}:{model_version}")
            
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to get cached model metadata for {model_name}:{model_version}: {e}")
            return None
    
    async def invalidate_transaction_cache(self, transaction: Transaction) -> int:
        """Invalidate all cached data for a transaction.
        
        Args:
            transaction: Transaction object
            
        Returns:
            Number of keys invalidated
        """
        try:
            await self._ensure_redis()
            
            tx_hash = self._generate_transaction_hash(transaction)
            
            # Patterns to clear
            patterns = [
                f"{self.prefixes['prediction']}*:{tx_hash}",
                f"{self.prefixes['features']}*:{tx_hash}"
            ]
            
            total_deleted = 0
            for pattern in patterns:
                deleted = await self.redis_manager.clear_pattern(pattern)
                total_deleted += deleted
            
            if total_deleted > 0:
                logger.info(f"Invalidated {total_deleted} cache entries for transaction {transaction.transaction_id}")
            
            return total_deleted
            
        except Exception as e:
            logger.error(f"Failed to invalidate cache for transaction {transaction.transaction_id}: {e}")
            return 0
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Cache statistics dictionary
        """
        try:
            await self._ensure_redis()
            
            redis_info = await self.redis_manager.get_connection_info()
            
            # Calculate hit rate
            hits = redis_info.get('keyspace_hits', 0)
            misses = redis_info.get('keyspace_misses', 0)
            total_requests = hits + misses
            hit_rate = (hits / total_requests * 100) if total_requests > 0 else 0
            
            return {
                'redis_info': redis_info,
                'hit_rate_percent': round(hit_rate, 2),
                'total_requests': total_requests,
                'cache_hits': hits,
                'cache_misses': misses
            }
            
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {'error': str(e)}


# Global ML cache service instance
_ml_cache_service: Optional[MLCacheService] = None


async def get_ml_cache_service() -> MLCacheService:
    """Get the global ML cache service instance.
    
    Returns:
        MLCacheService instance
    """
    global _ml_cache_service
    
    if _ml_cache_service is None:
        redis_manager = await get_redis_manager()
        _ml_cache_service = MLCacheService(redis_manager)
    
    return _ml_cache_service

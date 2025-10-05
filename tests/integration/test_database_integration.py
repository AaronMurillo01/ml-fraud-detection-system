"""Integration tests for database operations."""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import List, Dict, Any, Optional

from database.repositories import (
    TransactionRepository,
    UserRepository,
    MerchantRepository,
    FraudPredictionRepository,
    ModelRepository
)
from database.models import (
    TransactionModel,
    UserModel,
    MerchantModel,
    FraudPredictionModel,
    ModelMetadataModel
)
from shared.models import Transaction, FraudPrediction
from database.connection import DatabaseManager


@pytest.fixture
def mock_db_session():
    """Mock database session."""
    session = AsyncMock()
    session.add = Mock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.close = AsyncMock()
    session.execute = AsyncMock()
    session.scalar = AsyncMock()
    session.scalars = AsyncMock()
    return session


@pytest.fixture
def sample_transaction_model():
    """Sample transaction model for testing."""
    return TransactionModel(
        transaction_id="txn_123",
        user_id="user_123",
        merchant_id="merchant_123",
        amount=Decimal("150.75"),
        currency="USD",
        timestamp=datetime.now(timezone.utc),
        merchant_category="grocery",
        payment_method="credit_card",
        card_type="visa",
        location_data={
            "latitude": 40.7128,
            "longitude": -74.0060,
            "city": "New York",
            "country": "US"
        },
        created_at=datetime.now(timezone.utc)
    )


@pytest.fixture
def sample_user_model():
    """Sample user model for testing."""
    return UserModel(
        user_id="user_123",
        email="user@example.com",
        phone="+1234567890",
        registration_date=datetime.now(timezone.utc) - timedelta(days=30),
        kyc_status="verified",
        risk_level="low",
        profile_data={
            "age": 30,
            "occupation": "engineer",
            "income_range": "50k-100k"
        },
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc)
    )


@pytest.fixture
def sample_merchant_model():
    """Sample merchant model for testing."""
    return MerchantModel(
        merchant_id="merchant_123",
        name="Test Grocery Store",
        category="grocery",
        location_data={
            "address": "123 Main St",
            "city": "New York",
            "country": "US"
        },
        risk_score=0.1,
        registration_date=datetime.now(timezone.utc) - timedelta(days=365),
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc)
    )


@pytest.fixture
def sample_fraud_prediction_model():
    """Sample fraud prediction model for testing."""
    return FraudPredictionModel(
        prediction_id="pred_123",
        transaction_id="txn_123",
        user_id="user_123",
        is_fraud=True,
        fraud_probability=0.85,
        risk_score=0.92,
        confidence_score=0.88,
        model_version="xgb_v1.2.0",
        features_used=["amount_zscore", "frequency_1h", "location_risk"],
        prediction_timestamp=datetime.now(timezone.utc),
        created_at=datetime.now(timezone.utc)
    )


class TestDatabaseManager:
    """Test cases for database manager."""
    
    @patch('database.connection.create_async_engine')
    @patch('database.connection.async_sessionmaker')
    def test_database_manager_initialization(self, mock_sessionmaker, mock_engine):
        """Test database manager initialization."""
        mock_engine_instance = Mock()
        mock_engine.return_value = mock_engine_instance
        mock_sessionmaker.return_value = Mock()
        
        db_config = {
            "host": "localhost",
            "port": 5432,
            "database": "fraud_detection_test",
            "username": "test_user",
            "password": "test_pass"
        }
        
        db_manager = DatabaseManager(db_config)
        
        assert db_manager.config == db_config
        mock_engine.assert_called_once()
        mock_sessionmaker.assert_called_once_with(
            mock_engine_instance,
            expire_on_commit=False
        )
    
    @pytest.mark.asyncio
    @patch('database.connection.create_async_engine')
    @patch('database.connection.async_sessionmaker')
    async def test_get_session(self, mock_sessionmaker, mock_engine):
        """Test getting database session."""
        mock_session = AsyncMock()
        mock_session_factory = Mock(return_value=mock_session)
        mock_sessionmaker.return_value = mock_session_factory
        mock_engine.return_value = Mock()
        
        db_manager = DatabaseManager({})
        
        async with db_manager.get_session() as session:
            assert session == mock_session
        
        mock_session.close.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('database.connection.create_async_engine')
    async def test_health_check(self, mock_engine):
        """Test database health check."""
        mock_engine_instance = AsyncMock()
        mock_engine.return_value = mock_engine_instance
        
        db_manager = DatabaseManager({})
        
        # Mock successful connection
        mock_connection = AsyncMock()
        mock_engine_instance.connect.return_value.__aenter__.return_value = mock_connection
        
        health = await db_manager.health_check()
        
        assert health["status"] == "healthy"
        assert "response_time_ms" in health
        mock_engine_instance.connect.assert_called_once()


class TestTransactionRepository:
    """Test cases for transaction repository."""
    
    @pytest.mark.asyncio
    async def test_create_transaction(self, mock_db_session, sample_transaction_model):
        """Test creating a transaction."""
        repo = TransactionRepository(mock_db_session)
        
        result = await repo.create(sample_transaction_model)
        
        assert result == sample_transaction_model
        mock_db_session.add.assert_called_once_with(sample_transaction_model)
        mock_db_session.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_transaction_by_id(self, mock_db_session, sample_transaction_model):
        """Test getting transaction by ID."""
        mock_db_session.scalar.return_value = sample_transaction_model
        
        repo = TransactionRepository(mock_db_session)
        result = await repo.get_by_id("txn_123")
        
        assert result == sample_transaction_model
        mock_db_session.execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_transactions_by_user(self, mock_db_session, sample_transaction_model):
        """Test getting transactions by user ID."""
        mock_result = Mock()
        mock_result.scalars.return_value.all.return_value = [sample_transaction_model]
        mock_db_session.execute.return_value = mock_result
        
        repo = TransactionRepository(mock_db_session)
        result = await repo.get_by_user_id("user_123", limit=10)
        
        assert len(result) == 1
        assert result[0] == sample_transaction_model
        mock_db_session.execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_transactions_by_date_range(self, mock_db_session, sample_transaction_model):
        """Test getting transactions by date range."""
        mock_result = Mock()
        mock_result.scalars.return_value.all.return_value = [sample_transaction_model]
        mock_db_session.execute.return_value = mock_result
        
        repo = TransactionRepository(mock_db_session)
        start_date = datetime.now(timezone.utc) - timedelta(days=1)
        end_date = datetime.now(timezone.utc)
        
        result = await repo.get_by_date_range(start_date, end_date)
        
        assert len(result) == 1
        assert result[0] == sample_transaction_model
        mock_db_session.execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_update_transaction(self, mock_db_session, sample_transaction_model):
        """Test updating a transaction."""
        mock_db_session.scalar.return_value = sample_transaction_model
        
        repo = TransactionRepository(mock_db_session)
        updates = {"merchant_category": "restaurant"}
        
        result = await repo.update("txn_123", updates)
        
        assert result.merchant_category == "restaurant"
        mock_db_session.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_delete_transaction(self, mock_db_session, sample_transaction_model):
        """Test deleting a transaction."""
        mock_db_session.scalar.return_value = sample_transaction_model
        
        repo = TransactionRepository(mock_db_session)
        result = await repo.delete("txn_123")
        
        assert result is True
        mock_db_session.delete.assert_called_once_with(sample_transaction_model)
        mock_db_session.commit.assert_called_once()


class TestUserRepository:
    """Test cases for user repository."""
    
    @pytest.mark.asyncio
    async def test_create_user(self, mock_db_session, sample_user_model):
        """Test creating a user."""
        repo = UserRepository(mock_db_session)
        
        result = await repo.create(sample_user_model)
        
        assert result == sample_user_model
        mock_db_session.add.assert_called_once_with(sample_user_model)
        mock_db_session.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_user_by_id(self, mock_db_session, sample_user_model):
        """Test getting user by ID."""
        mock_db_session.scalar.return_value = sample_user_model
        
        repo = UserRepository(mock_db_session)
        result = await repo.get_by_id("user_123")
        
        assert result == sample_user_model
        mock_db_session.execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_user_by_email(self, mock_db_session, sample_user_model):
        """Test getting user by email."""
        mock_db_session.scalar.return_value = sample_user_model
        
        repo = UserRepository(mock_db_session)
        result = await repo.get_by_email("user@example.com")
        
        assert result == sample_user_model
        mock_db_session.execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_update_user_risk_level(self, mock_db_session, sample_user_model):
        """Test updating user risk level."""
        mock_db_session.scalar.return_value = sample_user_model
        
        repo = UserRepository(mock_db_session)
        result = await repo.update_risk_level("user_123", "high")
        
        assert result.risk_level == "high"
        mock_db_session.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_high_risk_users(self, mock_db_session, sample_user_model):
        """Test getting high-risk users."""
        sample_user_model.risk_level = "high"
        mock_result = Mock()
        mock_result.scalars.return_value.all.return_value = [sample_user_model]
        mock_db_session.execute.return_value = mock_result
        
        repo = UserRepository(mock_db_session)
        result = await repo.get_high_risk_users()
        
        assert len(result) == 1
        assert result[0].risk_level == "high"
        mock_db_session.execute.assert_called_once()


class TestFraudPredictionRepository:
    """Test cases for fraud prediction repository."""
    
    @pytest.mark.asyncio
    async def test_create_prediction(self, mock_db_session, sample_fraud_prediction_model):
        """Test creating a fraud prediction."""
        repo = FraudPredictionRepository(mock_db_session)
        
        result = await repo.create(sample_fraud_prediction_model)
        
        assert result == sample_fraud_prediction_model
        mock_db_session.add.assert_called_once_with(sample_fraud_prediction_model)
        mock_db_session.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_prediction_by_transaction_id(self, mock_db_session, sample_fraud_prediction_model):
        """Test getting prediction by transaction ID."""
        mock_db_session.scalar.return_value = sample_fraud_prediction_model
        
        repo = FraudPredictionRepository(mock_db_session)
        result = await repo.get_by_transaction_id("txn_123")
        
        assert result == sample_fraud_prediction_model
        mock_db_session.execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_fraud_predictions(self, mock_db_session, sample_fraud_prediction_model):
        """Test getting fraud predictions."""
        mock_result = Mock()
        mock_result.scalars.return_value.all.return_value = [sample_fraud_prediction_model]
        mock_db_session.execute.return_value = mock_result
        
        repo = FraudPredictionRepository(mock_db_session)
        result = await repo.get_fraud_predictions(limit=10)
        
        assert len(result) == 1
        assert result[0].is_fraud is True
        mock_db_session.execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_predictions_by_user(self, mock_db_session, sample_fraud_prediction_model):
        """Test getting predictions by user ID."""
        mock_result = Mock()
        mock_result.scalars.return_value.all.return_value = [sample_fraud_prediction_model]
        mock_db_session.execute.return_value = mock_result
        
        repo = FraudPredictionRepository(mock_db_session)
        result = await repo.get_by_user_id("user_123")
        
        assert len(result) == 1
        assert result[0].user_id == "user_123"
        mock_db_session.execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_prediction_statistics(self, mock_db_session):
        """Test getting prediction statistics."""
        mock_result = Mock()
        mock_result.fetchone.return_value = (100, 15, 0.15)  # total, fraud_count, fraud_rate
        mock_db_session.execute.return_value = mock_result
        
        repo = FraudPredictionRepository(mock_db_session)
        stats = await repo.get_statistics()
        
        assert stats["total_predictions"] == 100
        assert stats["fraud_predictions"] == 15
        assert stats["fraud_rate"] == 0.15
        mock_db_session.execute.assert_called_once()


class TestModelRepository:
    """Test cases for model repository."""
    
    @pytest.fixture
    def sample_model_metadata(self):
        """Sample model metadata for testing."""
        return ModelMetadataModel(
            model_id="model_123",
            name="XGBoost Fraud Detector",
            version="1.2.0",
            model_type="xgboost",
            training_date=datetime.now(timezone.utc),
            accuracy=0.95,
            precision=0.92,
            recall=0.88,
            f1_score=0.90,
            feature_names=["amount_zscore", "frequency_1h", "location_risk"],
            hyperparameters={
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1
            },
            is_active=True,
            created_at=datetime.now(timezone.utc)
        )
    
    @pytest.mark.asyncio
    async def test_create_model_metadata(self, mock_db_session, sample_model_metadata):
        """Test creating model metadata."""
        repo = ModelRepository(mock_db_session)
        
        result = await repo.create_metadata(sample_model_metadata)
        
        assert result == sample_model_metadata
        mock_db_session.add.assert_called_once_with(sample_model_metadata)
        mock_db_session.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_active_model(self, mock_db_session, sample_model_metadata):
        """Test getting active model."""
        mock_db_session.scalar.return_value = sample_model_metadata
        
        repo = ModelRepository(mock_db_session)
        result = await repo.get_active_model()
        
        assert result == sample_model_metadata
        assert result.is_active is True
        mock_db_session.execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_update_model_status(self, mock_db_session, sample_model_metadata):
        """Test updating model status."""
        mock_db_session.scalar.return_value = sample_model_metadata
        
        repo = ModelRepository(mock_db_session)
        result = await repo.update_status("model_123", False)
        
        assert result.is_active is False
        mock_db_session.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_model_performance_history(self, mock_db_session, sample_model_metadata):
        """Test getting model performance history."""
        mock_result = Mock()
        mock_result.scalars.return_value.all.return_value = [sample_model_metadata]
        mock_db_session.execute.return_value = mock_result
        
        repo = ModelRepository(mock_db_session)
        result = await repo.get_performance_history("model_123")
        
        assert len(result) == 1
        assert result[0].accuracy == 0.95
        mock_db_session.execute.assert_called_once()


class TestDatabaseIntegrationEnd2End:
    """End-to-end database integration tests."""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_transaction_fraud_prediction_workflow(self, mock_db_session, 
                                                       sample_transaction_model,
                                                       sample_fraud_prediction_model):
        """Test complete workflow from transaction to fraud prediction."""
        # Setup repositories
        transaction_repo = TransactionRepository(mock_db_session)
        prediction_repo = FraudPredictionRepository(mock_db_session)
        
        # Create transaction
        transaction = await transaction_repo.create(sample_transaction_model)
        assert transaction.transaction_id == "txn_123"
        
        # Create fraud prediction for the transaction
        prediction = await prediction_repo.create(sample_fraud_prediction_model)
        assert prediction.transaction_id == "txn_123"
        assert prediction.is_fraud is True
        
        # Verify both operations committed
        assert mock_db_session.commit.call_count == 2
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_user_risk_assessment_workflow(self, mock_db_session, 
                                                sample_user_model,
                                                sample_fraud_prediction_model):
        """Test user risk assessment workflow."""
        # Setup repositories
        user_repo = UserRepository(mock_db_session)
        prediction_repo = FraudPredictionRepository(mock_db_session)
        
        # Create user
        user = await user_repo.create(sample_user_model)
        assert user.risk_level == "low"
        
        # Simulate fraud prediction for user
        mock_result = Mock()
        mock_result.scalars.return_value.all.return_value = [sample_fraud_prediction_model]
        mock_db_session.execute.return_value = mock_result
        
        fraud_predictions = await prediction_repo.get_by_user_id("user_123")
        
        # Update user risk level based on fraud predictions
        if fraud_predictions and any(p.is_fraud for p in fraud_predictions):
            mock_db_session.scalar.return_value = user
            updated_user = await user_repo.update_risk_level("user_123", "high")
            assert updated_user.risk_level == "high"
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_database_error_handling(self, mock_db_session, sample_transaction_model):
        """Test database error handling and rollback."""
        # Setup repository
        repo = TransactionRepository(mock_db_session)
        
        # Simulate database error during commit
        mock_db_session.commit.side_effect = Exception("Database connection lost")
        
        with pytest.raises(Exception, match="Database connection lost"):
            await repo.create(sample_transaction_model)
        
        # Verify rollback was called
        mock_db_session.rollback.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
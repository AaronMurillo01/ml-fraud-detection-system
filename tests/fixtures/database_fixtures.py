"""Database fixtures for testing."""

import pytest
import asyncio
from typing import AsyncGenerator, List, Dict, Any
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime

from database.models import Base, UserModel, MerchantModel, FraudPredictionModel, ModelMetadataModel
from tests.fixtures.test_data import (
    sample_transactions, sample_users, sample_merchants, 
    sample_predictions, sample_model_metadata
)


# Test database URLs
TEST_DATABASE_URL = "sqlite:///./test_fraud_detection.db"
TEST_ASYNC_DATABASE_URL = "sqlite+aiosqlite:///./test_fraud_detection.db"
IN_MEMORY_DATABASE_URL = "sqlite:///:memory:"
IN_MEMORY_ASYNC_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="function")
def sync_engine():
    """Create a synchronous SQLAlchemy engine for testing."""
    engine = create_engine(
        IN_MEMORY_DATABASE_URL,
        connect_args={"check_same_thread": False},
        echo=False
    )
    
    # Create all tables
    Base.metadata.create_all(bind=engine)
    
    yield engine
    
    # Clean up
    Base.metadata.drop_all(bind=engine)
    engine.dispose()


@pytest.fixture(scope="function")
async def async_engine():
    """Create an asynchronous SQLAlchemy engine for testing."""
    engine = create_async_engine(
        IN_MEMORY_ASYNC_DATABASE_URL,
        echo=False
    )
    
    # Create all tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    yield engine
    
    # Clean up
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest.fixture(scope="function")
def sync_session(sync_engine) -> Session:
    """Create a synchronous database session for testing."""
    SessionLocal = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=sync_engine
    )
    
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture(scope="function")
async def async_session(async_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create an asynchronous database session for testing."""
    AsyncSessionLocal = async_sessionmaker(
        bind=async_engine,
        class_=AsyncSession,
        expire_on_commit=False
    )
    
    async with AsyncSessionLocal() as session:
        yield session


@pytest.fixture(scope="function")
def populated_sync_session(sync_session: Session) -> Session:
    """Create a populated synchronous database session with test data."""
    # Add sample users
    for user_data in sample_users:
        user = UserModel(
            user_id=user_data["user_id"],
            email=user_data["email"],
            phone_number=user_data["phone_number"],
            registration_date=user_data["registration_date"],
            last_login=user_data["last_login"],
            risk_score=user_data["risk_score"],
            status=user_data["status"],
            user_data=user_data["user_data"]
        )
        sync_session.add(user)
    
    # Add sample merchants
    for merchant_data in sample_merchants:
        merchant = MerchantModel(
            merchant_id=merchant_data["merchant_id"],
            merchant_name=merchant_data["merchant_name"],
            category=merchant_data["category"],
            location=merchant_data["location"],
            risk_score=merchant_data["risk_score"],
            status=merchant_data["status"],
            merchant_data=merchant_data["merchant_data"]
        )
        sync_session.add(merchant)
    
    # Add sample fraud predictions
    for prediction_data in sample_predictions:
        prediction = FraudPredictionModel(
            prediction_id=prediction_data["prediction_id"],
            transaction_id=prediction_data["transaction_id"],
            user_id=prediction_data["user_id"],
            fraud_probability=prediction_data["fraud_probability"],
            risk_level=prediction_data["risk_level"],
            decision=prediction_data["decision"],
            confidence_score=prediction_data["confidence_score"],
            model_version=prediction_data["model_version"],
            model_features=prediction_data["model_features"],
            feature_importance=prediction_data["feature_importance"],
            created_at=prediction_data["created_at"]
        )
        sync_session.add(prediction)
    
    # Add sample model metadata
    for model_data in sample_model_metadata:
        model_metadata = ModelMetadataModel(
            model_id=model_data["model_id"],
            model_name=model_data["model_name"],
            model_version=model_data["model_version"],
            model_type=model_data["model_type"],
            model_path=model_data["model_path"],
            feature_names=model_data["feature_names"],
            hyperparameters=model_data["hyperparameters"],
            performance_metrics=model_data["performance_metrics"],
            training_data_size=model_data["training_data_size"],
            validation_data_size=model_data["validation_data_size"],
            test_data_size=model_data["test_data_size"],
            deployment_date=model_data["deployment_date"],
            created_at=model_data["created_at"],
            updated_at=model_data["updated_at"]
        )
        sync_session.add(model_metadata)
    
    sync_session.commit()
    return sync_session


@pytest.fixture(scope="function")
async def populated_async_session(async_session: AsyncSession) -> AsyncSession:
    """Create a populated asynchronous database session with test data."""
    # Add sample users
    for user_data in sample_users:
        user = UserModel(
            user_id=user_data["user_id"],
            email=user_data["email"],
            phone_number=user_data["phone_number"],
            registration_date=user_data["registration_date"],
            last_login=user_data["last_login"],
            risk_score=user_data["risk_score"],
            status=user_data["status"],
            user_data=user_data["user_data"]
        )
        async_session.add(user)
    
    # Add sample merchants
    for merchant_data in sample_merchants:
        merchant = MerchantModel(
            merchant_id=merchant_data["merchant_id"],
            merchant_name=merchant_data["merchant_name"],
            category=merchant_data["category"],
            location=merchant_data["location"],
            risk_score=merchant_data["risk_score"],
            status=merchant_data["status"],
            merchant_data=merchant_data["merchant_data"]
        )
        async_session.add(merchant)
    
    # Add sample fraud predictions
    for prediction_data in sample_predictions:
        prediction = FraudPredictionModel(
            prediction_id=prediction_data["prediction_id"],
            transaction_id=prediction_data["transaction_id"],
            user_id=prediction_data["user_id"],
            fraud_probability=prediction_data["fraud_probability"],
            risk_level=prediction_data["risk_level"],
            decision=prediction_data["decision"],
            confidence_score=prediction_data["confidence_score"],
            model_version=prediction_data["model_version"],
            model_features=prediction_data["model_features"],
            feature_importance=prediction_data["feature_importance"],
            created_at=prediction_data["created_at"]
        )
        async_session.add(prediction)
    
    # Add sample model metadata
    for model_data in sample_model_metadata:
        model_metadata = ModelMetadataModel(
            model_id=model_data["model_id"],
            model_name=model_data["model_name"],
            model_version=model_data["model_version"],
            model_type=model_data["model_type"],
            model_path=model_data["model_path"],
            feature_names=model_data["feature_names"],
            hyperparameters=model_data["hyperparameters"],
            performance_metrics=model_data["performance_metrics"],
            training_data_size=model_data["training_data_size"],
            validation_data_size=model_data["validation_data_size"],
            test_data_size=model_data["test_data_size"],
            deployment_date=model_data["deployment_date"],
            created_at=model_data["created_at"],
            updated_at=model_data["updated_at"]
        )
        async_session.add(model_metadata)
    
    await async_session.commit()
    return async_session


@pytest.fixture(scope="function")
def clean_database(sync_session: Session):
    """Clean the database after each test."""
    yield
    
    # Clean up all tables
    sync_session.execute(text("DELETE FROM fraud_predictions"))
    sync_session.execute(text("DELETE FROM model_metadata"))
    sync_session.execute(text("DELETE FROM merchants"))
    sync_session.execute(text("DELETE FROM users"))
    sync_session.commit()


@pytest.fixture(scope="function")
async def clean_async_database(async_session: AsyncSession):
    """Clean the async database after each test."""
    yield
    
    # Clean up all tables
    await async_session.execute(text("DELETE FROM fraud_predictions"))
    await async_session.execute(text("DELETE FROM model_metadata"))
    await async_session.execute(text("DELETE FROM merchants"))
    await async_session.execute(text("DELETE FROM users"))
    await async_session.commit()


class DatabaseTestHelper:
    """Helper class for database testing operations."""
    
    def __init__(self, session: Session):
        self.session = session
    
    def create_user(self, **kwargs) -> UserModel:
        """Create a test user with default or custom values."""
        defaults = {
            "user_id": f"test_user_{datetime.utcnow().timestamp()}",
            "email": "test@example.com",
            "phone_number": "+1-555-0000",
            "registration_date": datetime.utcnow(),
            "last_login": datetime.utcnow(),
            "risk_score": 0.5,
            "status": "active",
            "user_data": {"test": True}
        }
        defaults.update(kwargs)
        
        user = UserModel(**defaults)
        self.session.add(user)
        self.session.commit()
        return user
    
    def create_merchant(self, **kwargs) -> MerchantModel:
        """Create a test merchant with default or custom values."""
        defaults = {
            "merchant_id": f"test_merchant_{datetime.utcnow().timestamp()}",
            "merchant_name": "Test Merchant",
            "category": "test",
            "location": "Test Location",
            "risk_score": 0.3,
            "status": "active",
            "merchant_data": {"test": True}
        }
        defaults.update(kwargs)
        
        merchant = MerchantModel(**defaults)
        self.session.add(merchant)
        self.session.commit()
        return merchant
    
    def create_fraud_prediction(self, **kwargs) -> FraudPredictionModel:
        """Create a test fraud prediction with default or custom values."""
        defaults = {
            "prediction_id": f"test_pred_{datetime.utcnow().timestamp()}",
            "transaction_id": f"test_txn_{datetime.utcnow().timestamp()}",
            "user_id": "test_user_001",
            "fraud_probability": 0.5,
            "risk_level": "MEDIUM",
            "decision": "REVIEW",
            "confidence_score": 0.8,
            "model_version": "test_v1.0.0",
            "model_features": {"test_feature": 1.0},
            "feature_importance": {"test_feature": 0.5},
            "created_at": datetime.utcnow()
        }
        defaults.update(kwargs)
        
        prediction = FraudPredictionModel(**defaults)
        self.session.add(prediction)
        self.session.commit()
        return prediction
    
    def create_model_metadata(self, **kwargs) -> ModelMetadataModel:
        """Create test model metadata with default or custom values."""
        defaults = {
            "model_id": f"test_model_{datetime.utcnow().timestamp()}",
            "model_name": "test_model",
            "model_version": "1.0.0",
            "model_type": "test",
            "model_path": "/test/model.pkl",
            "feature_names": ["feature1", "feature2"],
            "hyperparameters": {"param1": 1.0},
            "performance_metrics": {"accuracy": 0.9},
            "training_data_size": 1000,
            "validation_data_size": 200,
            "test_data_size": 100,
            "deployment_date": datetime.utcnow(),
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        defaults.update(kwargs)
        
        model_metadata = ModelMetadataModel(**defaults)
        self.session.add(model_metadata)
        self.session.commit()
        return model_metadata
    
    def count_records(self, model_class) -> int:
        """Count records in a table."""
        return self.session.query(model_class).count()
    
    def get_all_records(self, model_class) -> List:
        """Get all records from a table."""
        return self.session.query(model_class).all()
    
    def clear_table(self, model_class):
        """Clear all records from a table."""
        self.session.query(model_class).delete()
        self.session.commit()


class AsyncDatabaseTestHelper:
    """Helper class for async database testing operations."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def create_user(self, **kwargs) -> UserModel:
        """Create a test user with default or custom values."""
        defaults = {
            "user_id": f"test_user_{datetime.utcnow().timestamp()}",
            "email": "test@example.com",
            "phone_number": "+1-555-0000",
            "registration_date": datetime.utcnow(),
            "last_login": datetime.utcnow(),
            "risk_score": 0.5,
            "status": "active",
            "user_data": {"test": True}
        }
        defaults.update(kwargs)
        
        user = UserModel(**defaults)
        self.session.add(user)
        await self.session.commit()
        return user
    
    async def create_merchant(self, **kwargs) -> MerchantModel:
        """Create a test merchant with default or custom values."""
        defaults = {
            "merchant_id": f"test_merchant_{datetime.utcnow().timestamp()}",
            "merchant_name": "Test Merchant",
            "category": "test",
            "location": "Test Location",
            "risk_score": 0.3,
            "status": "active",
            "merchant_data": {"test": True}
        }
        defaults.update(kwargs)
        
        merchant = MerchantModel(**defaults)
        self.session.add(merchant)
        await self.session.commit()
        return merchant
    
    async def create_fraud_prediction(self, **kwargs) -> FraudPredictionModel:
        """Create a test fraud prediction with default or custom values."""
        defaults = {
            "prediction_id": f"test_pred_{datetime.utcnow().timestamp()}",
            "transaction_id": f"test_txn_{datetime.utcnow().timestamp()}",
            "user_id": "test_user_001",
            "fraud_probability": 0.5,
            "risk_level": "MEDIUM",
            "decision": "REVIEW",
            "confidence_score": 0.8,
            "model_version": "test_v1.0.0",
            "model_features": {"test_feature": 1.0},
            "feature_importance": {"test_feature": 0.5},
            "created_at": datetime.utcnow()
        }
        defaults.update(kwargs)
        
        prediction = FraudPredictionModel(**defaults)
        self.session.add(prediction)
        await self.session.commit()
        return prediction
    
    async def create_model_metadata(self, **kwargs) -> ModelMetadataModel:
        """Create test model metadata with default or custom values."""
        defaults = {
            "model_id": f"test_model_{datetime.utcnow().timestamp()}",
            "model_name": "test_model",
            "model_version": "1.0.0",
            "model_type": "test",
            "model_path": "/test/model.pkl",
            "feature_names": ["feature1", "feature2"],
            "hyperparameters": {"param1": 1.0},
            "performance_metrics": {"accuracy": 0.9},
            "training_data_size": 1000,
            "validation_data_size": 200,
            "test_data_size": 100,
            "deployment_date": datetime.utcnow(),
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        defaults.update(kwargs)
        
        model_metadata = ModelMetadataModel(**defaults)
        self.session.add(model_metadata)
        await self.session.commit()
        return model_metadata


@pytest.fixture(scope="function")
def db_helper(sync_session: Session) -> DatabaseTestHelper:
    """Provide a database test helper for synchronous operations."""
    return DatabaseTestHelper(sync_session)


@pytest.fixture(scope="function")
def async_db_helper(async_session: AsyncSession) -> AsyncDatabaseTestHelper:
    """Provide a database test helper for asynchronous operations."""
    return AsyncDatabaseTestHelper(async_session)


# Export fixtures and helpers
__all__ = [
    'sync_engine',
    'async_engine',
    'sync_session',
    'async_session',
    'populated_sync_session',
    'populated_async_session',
    'clean_database',
    'clean_async_database',
    'db_helper',
    'async_db_helper',
    'DatabaseTestHelper',
    'AsyncDatabaseTestHelper'
]
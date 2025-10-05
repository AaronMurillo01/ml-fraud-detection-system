"""Database repository classes."""

from typing import List, Optional, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete
from sqlalchemy.orm import selectinload
from datetime import datetime, timedelta

from .models import (
    TransactionModel,
    UserModel,
    MerchantModel,
    FraudPredictionModel,
    ModelMetadataModel
)


class BaseRepository:
    """Base repository class with common operations."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def commit(self):
        """Commit the current transaction."""
        await self.session.commit()
    
    async def rollback(self):
        """Rollback the current transaction."""
        await self.session.rollback()


class TransactionRepository(BaseRepository):
    """Repository for transaction operations."""
    
    async def create(self, transaction: TransactionModel) -> TransactionModel:
        """Create a new transaction."""
        self.session.add(transaction)
        await self.session.commit()
        return transaction
    
    async def get_by_id(self, transaction_id: str) -> Optional[TransactionModel]:
        """Get transaction by ID."""
        result = await self.session.execute(
            select(TransactionModel).where(TransactionModel.transaction_id == transaction_id)
        )
        return result.scalar_one_or_none()
    
    async def get_by_user_id(self, user_id: str, limit: int = 100) -> List[TransactionModel]:
        """Get transactions by user ID."""
        result = await self.session.execute(
            select(TransactionModel)
            .where(TransactionModel.user_id == user_id)
            .order_by(TransactionModel.timestamp.desc())
            .limit(limit)
        )
        return result.scalars().all()
    
    async def get_recent_transactions(self, user_id: str, hours: int = 24) -> List[TransactionModel]:
        """Get recent transactions for a user."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        result = await self.session.execute(
            select(TransactionModel)
            .where(
                TransactionModel.user_id == user_id,
                TransactionModel.timestamp >= cutoff_time
            )
            .order_by(TransactionModel.timestamp.desc())
        )
        return result.scalars().all()
    
    async def update(self, transaction_id: str, **kwargs) -> Optional[TransactionModel]:
        """Update transaction."""
        await self.session.execute(
            update(TransactionModel)
            .where(TransactionModel.transaction_id == transaction_id)
            .values(**kwargs)
        )
        await self.session.commit()
        return await self.get_by_id(transaction_id)
    
    async def delete(self, transaction_id: str) -> bool:
        """Delete transaction."""
        result = await self.session.execute(
            delete(TransactionModel)
            .where(TransactionModel.transaction_id == transaction_id)
        )
        await self.session.commit()
        return result.rowcount > 0


class UserRepository(BaseRepository):
    """Repository for user operations."""
    
    async def create(self, user: UserModel) -> UserModel:
        """Create a new user."""
        self.session.add(user)
        await self.session.commit()
        return user
    
    async def get_by_id(self, user_id: str) -> Optional[UserModel]:
        """Get user by ID."""
        result = await self.session.execute(
            select(UserModel).where(UserModel.user_id == user_id)
        )
        return result.scalar_one_or_none()
    
    async def get_by_email(self, email: str) -> Optional[UserModel]:
        """Get user by email."""
        result = await self.session.execute(
            select(UserModel).where(UserModel.email == email)
        )
        return result.scalar_one_or_none()
    
    async def update_risk_level(self, user_id: str, risk_level: str) -> Optional[UserModel]:
        """Update user risk level."""
        await self.session.execute(
            update(UserModel)
            .where(UserModel.user_id == user_id)
            .values(risk_level=risk_level, updated_at=datetime.utcnow())
        )
        await self.session.commit()
        return await self.get_by_id(user_id)


class MerchantRepository(BaseRepository):
    """Repository for merchant operations."""
    
    async def create(self, merchant: MerchantModel) -> MerchantModel:
        """Create a new merchant."""
        self.session.add(merchant)
        await self.session.commit()
        return merchant
    
    async def get_by_id(self, merchant_id: str) -> Optional[MerchantModel]:
        """Get merchant by ID."""
        result = await self.session.execute(
            select(MerchantModel).where(MerchantModel.merchant_id == merchant_id)
        )
        return result.scalar_one_or_none()
    
    async def get_by_category(self, category: str) -> List[MerchantModel]:
        """Get merchants by category."""
        result = await self.session.execute(
            select(MerchantModel).where(MerchantModel.category == category)
        )
        return result.scalars().all()
    
    async def update_risk_score(self, merchant_id: str, risk_score: float) -> Optional[MerchantModel]:
        """Update merchant risk score."""
        await self.session.execute(
            update(MerchantModel)
            .where(MerchantModel.merchant_id == merchant_id)
            .values(risk_score=risk_score, updated_at=datetime.utcnow())
        )
        await self.session.commit()
        return await self.get_by_id(merchant_id)


class FraudPredictionRepository(BaseRepository):
    """Repository for fraud prediction operations."""
    
    async def create(self, prediction: FraudPredictionModel) -> FraudPredictionModel:
        """Create a new fraud prediction."""
        self.session.add(prediction)
        await self.session.commit()
        return prediction
    
    async def get_by_id(self, prediction_id: str) -> Optional[FraudPredictionModel]:
        """Get prediction by ID."""
        result = await self.session.execute(
            select(FraudPredictionModel).where(FraudPredictionModel.prediction_id == prediction_id)
        )
        return result.scalar_one_or_none()
    
    async def get_by_transaction_id(self, transaction_id: str) -> Optional[FraudPredictionModel]:
        """Get prediction by transaction ID."""
        result = await self.session.execute(
            select(FraudPredictionModel).where(FraudPredictionModel.transaction_id == transaction_id)
        )
        return result.scalar_one_or_none()
    
    async def get_by_user_id(self, user_id: str, limit: int = 100) -> List[FraudPredictionModel]:
        """Get predictions by user ID."""
        result = await self.session.execute(
            select(FraudPredictionModel)
            .where(FraudPredictionModel.user_id == user_id)
            .order_by(FraudPredictionModel.prediction_timestamp.desc())
            .limit(limit)
        )
        return result.scalars().all()


class ModelRepository(BaseRepository):
    """Repository for model metadata operations."""
    
    async def create(self, model: ModelMetadataModel) -> ModelMetadataModel:
        """Create new model metadata."""
        self.session.add(model)
        await self.session.commit()
        return model
    
    async def get_by_id(self, model_id: str) -> Optional[ModelMetadataModel]:
        """Get model by ID."""
        result = await self.session.execute(
            select(ModelMetadataModel).where(ModelMetadataModel.model_id == model_id)
        )
        return result.scalar_one_or_none()
    
    async def get_active_models(self) -> List[ModelMetadataModel]:
        """Get all active models."""
        result = await self.session.execute(
            select(ModelMetadataModel)
            .where(ModelMetadataModel.is_active == True)
            .order_by(ModelMetadataModel.training_date.desc())
        )
        return result.scalars().all()
    
    async def get_latest_model(self, model_name: str) -> Optional[ModelMetadataModel]:
        """Get latest version of a model."""
        result = await self.session.execute(
            select(ModelMetadataModel)
            .where(ModelMetadataModel.model_name == model_name)
            .order_by(ModelMetadataModel.training_date.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()
    
    async def set_active(self, model_id: str) -> Optional[ModelMetadataModel]:
        """Set model as active and deactivate others."""
        # First deactivate all models with the same name
        model = await self.get_by_id(model_id)
        if model:
            await self.session.execute(
                update(ModelMetadataModel)
                .where(ModelMetadataModel.model_name == model.model_name)
                .values(is_active=False)
            )
            
            # Then activate the specified model
            await self.session.execute(
                update(ModelMetadataModel)
                .where(ModelMetadataModel.model_id == model_id)
                .values(is_active=True, updated_at=datetime.utcnow())
            )
            await self.session.commit()
            return await self.get_by_id(model_id)
        return None
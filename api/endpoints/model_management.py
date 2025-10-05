"""Model management API endpoints.

This module provides endpoints for:
- Model version management
- Model performance monitoring
- Model deployment and rollback
- Feature importance and explanations
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

from service.ml_inference import MLInferenceService, get_inference_service
from service.xgboost_model import create_xgboost_wrapper
from api.dependencies import (
    get_database,
    verify_api_key,
    get_current_user,
    require_admin_role
)
from config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(prefix="/models", tags=["model-management"])


# Request/Response Models
class ModelVersionInfo(BaseModel):
    """Model version information."""
    
    version: str = Field(..., description="Model version")
    model_type: str = Field(..., description="Model type (e.g., xgboost)")
    created_at: datetime = Field(..., description="Model creation timestamp")
    file_size_bytes: int = Field(..., description="Model file size in bytes")
    feature_count: int = Field(..., description="Number of features")
    performance_metrics: Dict[str, float] = Field(..., description="Model performance metrics")
    is_active: bool = Field(..., description="Whether this version is currently active")
    deployment_status: str = Field(..., description="Deployment status")
    
    class Config:
        schema_extra = {
            "example": {
                "version": "xgboost_v1.2.0",
                "model_type": "xgboost",
                "created_at": "2024-01-15T10:30:00Z",
                "file_size_bytes": 2048576,
                "feature_count": 45,
                "performance_metrics": {
                    "accuracy": 0.95,
                    "precision": 0.92,
                    "recall": 0.89,
                    "f1_score": 0.90,
                    "auc_roc": 0.96
                },
                "is_active": True,
                "deployment_status": "deployed"
            }
        }


class ModelPerformanceMetrics(BaseModel):
    """Model performance metrics."""
    
    model_version: str = Field(..., description="Model version")
    evaluation_date: datetime = Field(..., description="Evaluation timestamp")
    
    # Classification metrics
    accuracy: float = Field(..., description="Model accuracy")
    precision: float = Field(..., description="Model precision")
    recall: float = Field(..., description="Model recall")
    f1_score: float = Field(..., description="F1 score")
    auc_roc: float = Field(..., description="AUC-ROC score")
    auc_pr: float = Field(..., description="AUC-PR score")
    
    # Confusion matrix
    true_positives: int = Field(..., description="True positives")
    true_negatives: int = Field(..., description="True negatives")
    false_positives: int = Field(..., description="False positives")
    false_negatives: int = Field(..., description="False negatives")
    
    # Additional metrics
    log_loss: float = Field(..., description="Log loss")
    brier_score: float = Field(..., description="Brier score")
    
    class Config:
        schema_extra = {
            "example": {
                "model_version": "xgboost_v1.2.0",
                "evaluation_date": "2024-01-15T10:30:00Z",
                "accuracy": 0.95,
                "precision": 0.92,
                "recall": 0.89,
                "f1_score": 0.90,
                "auc_roc": 0.96,
                "auc_pr": 0.88,
                "true_positives": 450,
                "true_negatives": 9200,
                "false_positives": 80,
                "false_negatives": 270,
                "log_loss": 0.15,
                "brier_score": 0.08
            }
        }


class FeatureImportance(BaseModel):
    """Feature importance information."""
    
    feature_name: str = Field(..., description="Feature name")
    importance_score: float = Field(..., description="Importance score")
    importance_type: str = Field(..., description="Type of importance (gain, weight, cover)")
    rank: int = Field(..., description="Feature rank by importance")
    
    class Config:
        schema_extra = {
            "example": {
                "feature_name": "transaction_amount_zscore",
                "importance_score": 0.15,
                "importance_type": "gain",
                "rank": 1
            }
        }


class ModelDeploymentRequest(BaseModel):
    """Model deployment request."""
    
    model_version: str = Field(..., description="Model version to deploy")
    deployment_strategy: str = Field(default="immediate", description="Deployment strategy")
    rollback_on_error: bool = Field(default=True, description="Rollback on deployment error")
    health_check_timeout: int = Field(default=300, description="Health check timeout in seconds")
    
    @validator('deployment_strategy')
    def validate_deployment_strategy(cls, v):
        allowed_strategies = ["immediate", "canary", "blue_green"]
        if v not in allowed_strategies:
            raise ValueError(f"Deployment strategy must be one of: {allowed_strategies}")
        return v


class ModelDeploymentResponse(BaseModel):
    """Model deployment response."""
    
    deployment_id: str = Field(..., description="Deployment ID")
    model_version: str = Field(..., description="Deployed model version")
    deployment_status: str = Field(..., description="Deployment status")
    deployment_strategy: str = Field(..., description="Deployment strategy used")
    started_at: datetime = Field(..., description="Deployment start time")
    completed_at: Optional[datetime] = Field(default=None, description="Deployment completion time")
    message: str = Field(..., description="Deployment message")
    
    class Config:
        schema_extra = {
            "example": {
                "deployment_id": "deploy_20240115_103000",
                "model_version": "xgboost_v1.2.0",
                "deployment_status": "completed",
                "deployment_strategy": "immediate",
                "started_at": "2024-01-15T10:30:00Z",
                "completed_at": "2024-01-15T10:30:15Z",
                "message": "Model deployed successfully"
            }
        }


# Endpoints
@router.get(
    "",
    summary="Get models overview",
    description="Get overview of all models and their status."
)
async def get_models_overview(
    inference_service: MLInferenceService = Depends(get_inference_service)
):
    """Get overview of all models."""
    try:
        return {
            "models": [
                {
                    "id": "fraud_detector_xgb",
                    "name": "XGBoost Fraud Detector",
                    "version": "v1.2.0",
                    "type": "xgboost",
                    "status": "active",
                    "accuracy": 0.95,
                    "precision": 0.92,
                    "recall": 0.89,
                    "f1_score": 0.90,
                    "last_trained": "2024-01-15T10:30:00Z",
                    "predictions_count": 125000
                },
                {
                    "id": "fraud_detector_rf",
                    "name": "Random Forest Fraud Detector",
                    "version": "v1.1.0",
                    "type": "random_forest",
                    "status": "inactive",
                    "accuracy": 0.93,
                    "precision": 0.90,
                    "recall": 0.87,
                    "f1_score": 0.88,
                    "last_trained": "2024-01-10T08:15:00Z",
                    "predictions_count": 98000
                },
                {
                    "id": "fraud_detector_nn",
                    "name": "Neural Network Fraud Detector",
                    "version": "v1.0.0",
                    "type": "neural_network",
                    "status": "training",
                    "accuracy": 0.91,
                    "precision": 0.88,
                    "recall": 0.85,
                    "f1_score": 0.86,
                    "last_trained": "2024-01-05T14:20:00Z",
                    "predictions_count": 75000
                }
            ],
            "total_models": 3,
            "active_models": 1,
            "total_predictions": 298000
        }
    except Exception as e:
        logger.error(f"Error getting models overview: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get models overview")


@router.get(
    "/versions",
    response_model=List[ModelVersionInfo],
    summary="List model versions",
    description="Get list of all available model versions with their information."
)
async def list_model_versions(
    limit: int = Query(default=10, ge=1, le=100, description="Maximum number of versions to return"),
    offset: int = Query(default=0, ge=0, description="Number of versions to skip"),
    include_inactive: bool = Query(default=False, description="Include inactive model versions"),
    db = Depends(get_database),
    api_key: str = Depends(verify_api_key)
):
    """List available model versions.
    
    Args:
        limit: Maximum number of versions to return
        offset: Number of versions to skip
        include_inactive: Include inactive model versions
        db: Database connection
        api_key: API key for authentication
        
    Returns:
        List of model version information
    """
    try:
        # Query model versions from database
        query = """
        SELECT 
            version, model_type, created_at, file_size_bytes,
            feature_count, performance_metrics, is_active, deployment_status
        FROM model_metadata
        """
        
        conditions = []
        params = []
        
        if not include_inactive:
            conditions.append("is_active = $1")
            params.append(True)
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY created_at DESC LIMIT $" + str(len(params) + 1) + " OFFSET $" + str(len(params) + 2)
        params.extend([limit, offset])
        
        rows = await db.fetch(query, *params)
        
        versions = []
        for row in rows:
            version_info = ModelVersionInfo(
                version=row['version'],
                model_type=row['model_type'],
                created_at=row['created_at'],
                file_size_bytes=row['file_size_bytes'],
                feature_count=row['feature_count'],
                performance_metrics=row['performance_metrics'],
                is_active=row['is_active'],
                deployment_status=row['deployment_status']
            )
            versions.append(version_info)
        
        logger.info(f"Retrieved {len(versions)} model versions")
        return versions
        
    except Exception as e:
        logger.error(f"Failed to list model versions: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "model_versions_failed",
                "message": "Failed to retrieve model versions"
            }
        )


@router.get(
    "/versions/{version}/performance",
    response_model=ModelPerformanceMetrics,
    summary="Get model performance metrics",
    description="Get detailed performance metrics for a specific model version."
)
async def get_model_performance(
    version: str,
    db = Depends(get_database),
    api_key: str = Depends(verify_api_key)
):
    """Get model performance metrics.
    
    Args:
        version: Model version
        db: Database connection
        api_key: API key for authentication
        
    Returns:
        Model performance metrics
    """
    try:
        # Query performance metrics from database
        query = """
        SELECT 
            model_version, evaluation_date, accuracy, precision, recall,
            f1_score, auc_roc, auc_pr, true_positives, true_negatives,
            false_positives, false_negatives, log_loss, brier_score
        FROM model_performance_metrics
        WHERE model_version = $1
        ORDER BY evaluation_date DESC
        LIMIT 1
        """
        
        row = await db.fetchrow(query, version)
        
        if not row:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "model_not_found",
                    "message": f"Performance metrics not found for model version {version}"
                }
            )
        
        metrics = ModelPerformanceMetrics(
            model_version=row['model_version'],
            evaluation_date=row['evaluation_date'],
            accuracy=row['accuracy'],
            precision=row['precision'],
            recall=row['recall'],
            f1_score=row['f1_score'],
            auc_roc=row['auc_roc'],
            auc_pr=row['auc_pr'],
            true_positives=row['true_positives'],
            true_negatives=row['true_negatives'],
            false_positives=row['false_positives'],
            false_negatives=row['false_negatives'],
            log_loss=row['log_loss'],
            brier_score=row['brier_score']
        )
        
        logger.info(f"Retrieved performance metrics for model {version}")
        return metrics
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get model performance for {version}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "model_performance_failed",
                "message": "Failed to retrieve model performance metrics"
            }
        )


@router.get(
    "/versions/{version}/features",
    response_model=List[FeatureImportance],
    summary="Get feature importance",
    description="Get feature importance scores for a specific model version."
)
async def get_feature_importance(
    version: str,
    importance_type: str = Query(default="gain", description="Type of importance (gain, weight, cover)"),
    top_k: int = Query(default=20, ge=1, le=100, description="Number of top features to return"),
    ml_service: MLInferenceService = Depends(get_inference_service),
    api_key: str = Depends(verify_api_key)
):
    """Get feature importance for a model version.
    
    Args:
        version: Model version
        importance_type: Type of importance to retrieve
        top_k: Number of top features to return
        ml_service: ML inference service
        api_key: API key for authentication
        
    Returns:
        List of feature importance scores
    """
    try:
        # Get XGBoost wrapper for the specific version
        xgb_wrapper = await get_xgboost_wrapper(version)
        
        if not xgb_wrapper:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "model_not_found",
                    "message": f"Model version {version} not found"
                }
            )
        
        # Get feature importance
        importance_scores = await xgb_wrapper.get_feature_importance(importance_type)
        
        # Sort by importance and take top_k
        sorted_features = sorted(
            importance_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        # Build response
        feature_importance = []
        for rank, (feature_name, score) in enumerate(sorted_features, 1):
            importance = FeatureImportance(
                feature_name=feature_name,
                importance_score=score,
                importance_type=importance_type,
                rank=rank
            )
            feature_importance.append(importance)
        
        logger.info(f"Retrieved {len(feature_importance)} feature importance scores for model {version}")
        return feature_importance
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get feature importance for {version}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "feature_importance_failed",
                "message": "Failed to retrieve feature importance"
            }
        )


@router.post(
    "/deploy",
    response_model=ModelDeploymentResponse,
    summary="Deploy model version",
    description="Deploy a specific model version to production."
)
async def deploy_model(
    request: ModelDeploymentRequest,
    background_tasks: BackgroundTasks,
    ml_service: MLInferenceService = Depends(get_inference_service),
    db = Depends(get_database),
    api_key: str = Depends(verify_api_key),
    current_user = Depends(get_current_user),
    admin_user = Depends(require_admin_role)
):
    """Deploy a model version to production.
    
    Args:
        request: Model deployment request
        background_tasks: FastAPI background tasks
        ml_service: ML inference service
        db: Database connection
        api_key: API key for authentication
        current_user: Current authenticated user
        admin_user: Admin user verification
        
    Returns:
        Model deployment response
    """
    deployment_id = f"deploy_{int(datetime.now().timestamp())}"
    
    try:
        logger.info(
            f"Starting model deployment - ID: {deployment_id} - "
            f"Version: {request.model_version} - "
            f"Strategy: {request.deployment_strategy} - "
            f"User: {current_user}"
        )
        
        # Check if model version exists
        version_query = "SELECT version FROM model_metadata WHERE version = $1 AND is_active = true"
        version_exists = await db.fetchrow(version_query, request.model_version)
        
        if not version_exists:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "model_version_not_found",
                    "message": f"Model version {request.model_version} not found or inactive"
                }
            )
        
        # Create deployment record
        deployment_query = """
        INSERT INTO model_deployments (
            deployment_id, model_version, deployment_strategy,
            deployment_status, started_at, started_by
        ) VALUES ($1, $2, $3, $4, $5, $6)
        """
        
        await db.execute(
            deployment_query,
            deployment_id,
            request.model_version,
            request.deployment_strategy,
            "in_progress",
            datetime.now(),
            current_user
        )
        
        # Start deployment in background
        background_tasks.add_task(
            execute_model_deployment,
            deployment_id,
            request,
            ml_service,
            db
        )
        
        response = ModelDeploymentResponse(
            deployment_id=deployment_id,
            model_version=request.model_version,
            deployment_status="in_progress",
            deployment_strategy=request.deployment_strategy,
            started_at=datetime.now(),
            message="Model deployment started"
        )
        
        logger.info(f"Model deployment initiated - ID: {deployment_id}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start model deployment: {e}", exc_info=True)
        
        # Update deployment status to failed
        try:
            await db.execute(
                "UPDATE model_deployments SET deployment_status = $1, error_message = $2 WHERE deployment_id = $3",
                "failed", str(e), deployment_id
            )
        except Exception:
            pass
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "model_deployment_failed",
                "message": "Failed to start model deployment",
                "deployment_id": deployment_id
            }
        )


@router.get(
    "/deployments/{deployment_id}",
    response_model=ModelDeploymentResponse,
    summary="Get deployment status",
    description="Get the status of a model deployment."
)
async def get_deployment_status(
    deployment_id: str,
    db = Depends(get_database),
    api_key: str = Depends(verify_api_key)
):
    """Get deployment status.
    
    Args:
        deployment_id: Deployment ID
        db: Database connection
        api_key: API key for authentication
        
    Returns:
        Model deployment response
    """
    try:
        query = """
        SELECT 
            deployment_id, model_version, deployment_strategy,
            deployment_status, started_at, completed_at, error_message
        FROM model_deployments
        WHERE deployment_id = $1
        """
        
        row = await db.fetchrow(query, deployment_id)
        
        if not row:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "deployment_not_found",
                    "message": f"Deployment {deployment_id} not found"
                }
            )
        
        # Determine message based on status
        status = row['deployment_status']
        if status == "completed":
            message = "Model deployed successfully"
        elif status == "failed":
            message = f"Deployment failed: {row['error_message'] or 'Unknown error'}"
        elif status == "in_progress":
            message = "Deployment in progress"
        else:
            message = f"Deployment status: {status}"
        
        response = ModelDeploymentResponse(
            deployment_id=row['deployment_id'],
            model_version=row['model_version'],
            deployment_status=row['deployment_status'],
            deployment_strategy=row['deployment_strategy'],
            started_at=row['started_at'],
            completed_at=row['completed_at'],
            message=message
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get deployment status for {deployment_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "deployment_status_failed",
                "message": "Failed to retrieve deployment status"
            }
        )


# Background tasks
async def execute_model_deployment(
    deployment_id: str,
    request: ModelDeploymentRequest,
    ml_service: MLInferenceService,
    db
):
    """Execute model deployment in background.
    
    Args:
        deployment_id: Deployment ID
        request: Deployment request
        ml_service: ML inference service
        db: Database connection
    """
    try:
        logger.info(f"Executing model deployment - ID: {deployment_id}")
        
        # Load the new model
        await ml_service.reload_model(request.model_version)
        
        # Perform health check
        health_check_start = datetime.now()
        while (datetime.now() - health_check_start).seconds < request.health_check_timeout:
            try:
                status = await ml_service.get_model_status()
                if status["loaded"] and status["version"] == request.model_version:
                    break
            except Exception:
                pass
            
            await asyncio.sleep(5)  # Wait 5 seconds before next check
        else:
            raise Exception("Health check timeout - model not ready")
        
        # Update deployment status to completed
        await db.execute(
            "UPDATE model_deployments SET deployment_status = $1, completed_at = $2 WHERE deployment_id = $3",
            "completed", datetime.now(), deployment_id
        )
        
        # Update model metadata
        await db.execute(
            "UPDATE model_metadata SET deployment_status = $1 WHERE version = $2",
            "deployed", request.model_version
        )
        
        logger.info(f"Model deployment completed successfully - ID: {deployment_id}")
        
    except Exception as e:
        logger.error(f"Model deployment failed - ID: {deployment_id} - Error: {e}", exc_info=True)
        
        # Update deployment status to failed
        try:
            await db.execute(
                "UPDATE model_deployments SET deployment_status = $1, completed_at = $2, error_message = $3 WHERE deployment_id = $4",
                "failed", datetime.now(), str(e), deployment_id
            )
        except Exception as update_error:
            logger.error(f"Failed to update deployment status: {update_error}")
        
        # Rollback if requested
        if request.rollback_on_error:
            try:
                # Get previous active model
                previous_model_query = """
                SELECT version FROM model_metadata 
                WHERE deployment_status = 'deployed' AND version != $1
                ORDER BY created_at DESC LIMIT 1
                """
                
                previous_model = await db.fetchrow(previous_model_query, request.model_version)
                
                if previous_model:
                    await ml_service.reload_model(previous_model['version'])
                    logger.info(f"Rolled back to previous model version: {previous_model['version']}")
                
            except Exception as rollback_error:
                logger.error(f"Rollback failed: {rollback_error}", exc_info=True)
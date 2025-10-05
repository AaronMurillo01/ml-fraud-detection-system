"""Batch processing endpoints for fraud detection.

This module provides endpoints for:
- Batch transaction upload (CSV, Excel, JSON)
- Batch fraud scoring
- Batch result export
- Processing status tracking
"""

import logging
import asyncio
import csv
import io
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum

from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, BackgroundTasks, Query
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field, validator
import pandas as pd

from service.ml_inference import MLInferenceService, get_inference_service
from features.feature_pipeline import FeaturePipeline, get_feature_pipeline
from api.auth import get_current_active_user, User
from api.dependencies import get_database

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/batch", tags=["Batch Processing"])


class BatchStatus(str, Enum):
    """Batch processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


class BatchJobRequest(BaseModel):
    """Request model for batch job creation."""
    name: Optional[str] = Field(None, description="Job name")
    description: Optional[str] = Field(None, description="Job description")
    notify_on_completion: bool = Field(False, description="Send notification when complete")


class BatchJobResponse(BaseModel):
    """Response model for batch job."""
    job_id: str = Field(..., description="Unique job identifier")
    status: BatchStatus = Field(..., description="Job status")
    name: Optional[str] = Field(None, description="Job name")
    total_records: int = Field(..., description="Total records to process")
    processed_records: int = Field(0, description="Records processed")
    successful_records: int = Field(0, description="Successfully processed records")
    failed_records: int = Field(0, description="Failed records")
    created_at: datetime = Field(..., description="Job creation time")
    started_at: Optional[datetime] = Field(None, description="Processing start time")
    completed_at: Optional[datetime] = Field(None, description="Processing completion time")
    results_url: Optional[str] = Field(None, description="URL to download results")
    error_message: Optional[str] = Field(None, description="Error message if failed")


class BatchResult(BaseModel):
    """Individual batch result."""
    transaction_id: str
    fraud_score: float
    is_fraud: bool
    risk_level: str
    processing_time_ms: float
    error: Optional[str] = None


# In-memory job store (use Redis or database in production)
batch_jobs: Dict[str, Dict[str, Any]] = {}


def parse_csv_file(file_content: bytes) -> List[Dict[str, Any]]:
    """Parse CSV file content into list of transactions."""
    try:
        df = pd.read_csv(io.BytesIO(file_content))
        return df.to_dict('records')
    except Exception as e:
        logger.error(f"Error parsing CSV: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid CSV format: {str(e)}")


def parse_excel_file(file_content: bytes) -> List[Dict[str, Any]]:
    """Parse Excel file content into list of transactions."""
    try:
        df = pd.read_excel(io.BytesIO(file_content))
        return df.to_dict('records')
    except Exception as e:
        logger.error(f"Error parsing Excel: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid Excel format: {str(e)}")


def parse_json_file(file_content: bytes) -> List[Dict[str, Any]]:
    """Parse JSON file content into list of transactions."""
    try:
        import json
        data = json.loads(file_content.decode('utf-8'))
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and 'transactions' in data:
            return data['transactions']
        else:
            raise ValueError("JSON must be a list or contain 'transactions' key")
    except Exception as e:
        logger.error(f"Error parsing JSON: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid JSON format: {str(e)}")


async def process_batch_job(
    job_id: str,
    transactions: List[Dict[str, Any]],
    ml_service: MLInferenceService,
    feature_pipeline: FeaturePipeline
):
    """Process batch job in background."""
    job = batch_jobs[job_id]
    
    try:
        job["status"] = BatchStatus.PROCESSING
        job["started_at"] = datetime.utcnow()
        
        results = []
        
        for idx, transaction in enumerate(transactions):
            try:
                start_time = datetime.utcnow()
                
                # Process transaction
                # Note: This is a simplified version. In production, use proper feature extraction
                features = await feature_pipeline.extract_features(transaction)
                prediction = await ml_service.predict(features)
                
                processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                
                result = BatchResult(
                    transaction_id=transaction.get('transaction_id', f'txn_{idx}'),
                    fraud_score=prediction.get('fraud_score', 0.0),
                    is_fraud=prediction.get('is_fraud', False),
                    risk_level=prediction.get('risk_level', 'low'),
                    processing_time_ms=processing_time
                )
                
                results.append(result.dict())
                job["successful_records"] += 1
                
            except Exception as e:
                logger.error(f"Error processing transaction {idx}: {e}")
                result = BatchResult(
                    transaction_id=transaction.get('transaction_id', f'txn_{idx}'),
                    fraud_score=0.0,
                    is_fraud=False,
                    risk_level='unknown',
                    processing_time_ms=0.0,
                    error=str(e)
                )
                results.append(result.dict())
                job["failed_records"] += 1
            
            job["processed_records"] = idx + 1
        
        # Store results
        job["results"] = results
        job["status"] = BatchStatus.COMPLETED if job["failed_records"] == 0 else BatchStatus.PARTIAL
        job["completed_at"] = datetime.utcnow()
        job["results_url"] = f"/api/v1/batch/{job_id}/results"
        
        logger.info(f"Batch job {job_id} completed: {job['successful_records']}/{job['total_records']} successful")
        
    except Exception as e:
        logger.error(f"Batch job {job_id} failed: {e}")
        job["status"] = BatchStatus.FAILED
        job["error_message"] = str(e)
        job["completed_at"] = datetime.utcnow()


@router.post("/upload", response_model=BatchJobResponse)
async def upload_batch_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    job_request: Optional[BatchJobRequest] = None,
    ml_service: MLInferenceService = Depends(get_inference_service),
    feature_pipeline: FeaturePipeline = Depends(get_feature_pipeline),
    current_user: User = Depends(get_current_active_user)
):
    """Upload a batch file for fraud detection processing.
    
    Supported formats: CSV, Excel (.xlsx, .xls), JSON
    
    Args:
        file: Upload file
        job_request: Optional job configuration
        ml_service: ML inference service
        feature_pipeline: Feature pipeline
        current_user: Current authenticated user
    
    Returns:
        Batch job information
    """
    
    # Validate file type
    allowed_extensions = ['.csv', '.xlsx', '.xls', '.json']
    file_ext = file.filename.split('.')[-1].lower()
    
    if f'.{file_ext}' not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
        )
    
    # Read file content
    file_content = await file.read()
    
    # Parse file based on type
    if file_ext == 'csv':
        transactions = parse_csv_file(file_content)
    elif file_ext in ['xlsx', 'xls']:
        transactions = parse_excel_file(file_content)
    elif file_ext == 'json':
        transactions = parse_json_file(file_content)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file format")
    
    if not transactions:
        raise HTTPException(status_code=400, detail="No transactions found in file")
    
    if len(transactions) > 10000:
        raise HTTPException(
            status_code=400,
            detail="Maximum 10,000 transactions per batch. Please split your file."
        )
    
    # Create batch job
    job_id = str(uuid.uuid4())
    job = {
        "job_id": job_id,
        "status": BatchStatus.PENDING,
        "name": job_request.name if job_request else file.filename,
        "total_records": len(transactions),
        "processed_records": 0,
        "successful_records": 0,
        "failed_records": 0,
        "created_at": datetime.utcnow(),
        "started_at": None,
        "completed_at": None,
        "results_url": None,
        "error_message": None,
        "user_id": current_user.id,
        "results": []
    }
    
    batch_jobs[job_id] = job
    
    # Start background processing
    background_tasks.add_task(
        process_batch_job,
        job_id,
        transactions,
        ml_service,
        feature_pipeline
    )
    
    logger.info(f"Created batch job {job_id} with {len(transactions)} transactions")
    
    return BatchJobResponse(**job)


@router.get("/{job_id}", response_model=BatchJobResponse)
async def get_batch_job_status(
    job_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """Get batch job status.
    
    Args:
        job_id: Batch job ID
        current_user: Current authenticated user
    
    Returns:
        Batch job status
    """
    if job_id not in batch_jobs:
        raise HTTPException(status_code=404, detail="Batch job not found")
    
    job = batch_jobs[job_id]
    
    # Check authorization
    if job["user_id"] != current_user.id and current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Not authorized to access this job")
    
    return BatchJobResponse(**job)


@router.get("/{job_id}/results")
async def get_batch_results(
    job_id: str,
    format: str = Query("json", regex="^(json|csv)$"),
    current_user: User = Depends(get_current_active_user)
):
    """Download batch processing results.
    
    Args:
        job_id: Batch job ID
        format: Output format (json or csv)
        current_user: Current authenticated user
    
    Returns:
        Batch results in requested format
    """
    if job_id not in batch_jobs:
        raise HTTPException(status_code=404, detail="Batch job not found")
    
    job = batch_jobs[job_id]
    
    # Check authorization
    if job["user_id"] != current_user.id and current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Not authorized to access this job")
    
    if job["status"] not in [BatchStatus.COMPLETED, BatchStatus.PARTIAL]:
        raise HTTPException(status_code=400, detail="Job not completed yet")
    
    results = job.get("results", [])
    
    if format == "csv":
        # Generate CSV
        output = io.StringIO()
        if results:
            writer = csv.DictWriter(output, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=batch_results_{job_id}.csv"}
        )
    else:
        # Return JSON
        return JSONResponse(content={"job_id": job_id, "results": results})


@router.get("/")
async def list_batch_jobs(
    status: Optional[BatchStatus] = None,
    limit: int = Query(50, ge=1, le=100),
    current_user: User = Depends(get_current_active_user)
):
    """List batch jobs for current user.
    
    Args:
        status: Filter by status
        limit: Maximum number of jobs to return
        current_user: Current authenticated user
    
    Returns:
        List of batch jobs
    """
    user_jobs = [
        BatchJobResponse(**job)
        for job in batch_jobs.values()
        if job["user_id"] == current_user.id or current_user.role == "admin"
    ]
    
    if status:
        user_jobs = [job for job in user_jobs if job.status == status]
    
    # Sort by creation time (newest first)
    user_jobs.sort(key=lambda x: x.created_at, reverse=True)
    
    return {"jobs": user_jobs[:limit], "total": len(user_jobs)}


@router.delete("/{job_id}")
async def delete_batch_job(
    job_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """Delete a batch job.
    
    Args:
        job_id: Batch job ID
        current_user: Current authenticated user
    
    Returns:
        Success message
    """
    if job_id not in batch_jobs:
        raise HTTPException(status_code=404, detail="Batch job not found")
    
    job = batch_jobs[job_id]
    
    # Check authorization
    if job["user_id"] != current_user.id and current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Not authorized to delete this job")
    
    del batch_jobs[job_id]
    
    logger.info(f"Deleted batch job {job_id}")
    
    return {"message": "Batch job deleted successfully", "job_id": job_id}


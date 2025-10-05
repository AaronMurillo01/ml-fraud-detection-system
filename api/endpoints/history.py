"""Transaction history and search endpoints.

This module provides endpoints for:
- Transaction history viewing
- Advanced search and filtering
- Transaction details
- Fraud investigation tools
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from enum import Enum

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from api.auth import get_current_active_user, User, UserRole
from api.dependencies import get_database

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/history", tags=["Transaction History"])


class RiskLevel(str, Enum):
    """Risk level classification."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TransactionStatus(str, Enum):
    """Transaction status."""
    APPROVED = "approved"
    DECLINED = "declined"
    PENDING = "pending"
    UNDER_REVIEW = "under_review"


class SortField(str, Enum):
    """Fields available for sorting."""
    TIMESTAMP = "timestamp"
    AMOUNT = "amount"
    FRAUD_SCORE = "fraud_score"
    RISK_LEVEL = "risk_level"


class SortOrder(str, Enum):
    """Sort order."""
    ASC = "asc"
    DESC = "desc"


class TransactionDetail(BaseModel):
    """Detailed transaction information."""
    transaction_id: str = Field(..., description="Unique transaction identifier")
    user_id: str = Field(..., description="User identifier")
    merchant_id: str = Field(..., description="Merchant identifier")
    merchant_name: str = Field(..., description="Merchant name")
    amount: float = Field(..., description="Transaction amount")
    currency: str = Field(..., description="Currency code")
    timestamp: datetime = Field(..., description="Transaction timestamp")
    
    # Fraud detection results
    fraud_score: float = Field(..., description="Fraud probability score (0-1)")
    is_fraud: bool = Field(..., description="Fraud classification")
    risk_level: RiskLevel = Field(..., description="Risk level")
    status: TransactionStatus = Field(..., description="Transaction status")
    
    # Additional details
    ip_address: Optional[str] = Field(None, description="IP address")
    device_id: Optional[str] = Field(None, description="Device identifier")
    location: Optional[Dict[str, Any]] = Field(None, description="Transaction location")
    
    # Risk factors
    risk_factors: List[str] = Field(default_factory=list, description="Identified risk factors")
    
    # Investigation
    reviewed_by: Optional[str] = Field(None, description="Reviewer username")
    reviewed_at: Optional[datetime] = Field(None, description="Review timestamp")
    notes: Optional[str] = Field(None, description="Investigation notes")


class TransactionSummary(BaseModel):
    """Summary transaction information for list view."""
    transaction_id: str
    user_id: str
    merchant_name: str
    amount: float
    currency: str
    timestamp: datetime
    fraud_score: float
    risk_level: RiskLevel
    status: TransactionStatus


class SearchFilters(BaseModel):
    """Search and filter parameters."""
    user_id: Optional[str] = None
    merchant_id: Optional[str] = None
    min_amount: Optional[float] = None
    max_amount: Optional[float] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    risk_levels: Optional[List[RiskLevel]] = None
    statuses: Optional[List[TransactionStatus]] = None
    min_fraud_score: Optional[float] = None
    max_fraud_score: Optional[float] = None
    is_fraud: Optional[bool] = None
    search_query: Optional[str] = None


class PaginatedResponse(BaseModel):
    """Paginated response model."""
    items: List[TransactionSummary]
    total: int
    page: int
    page_size: int
    total_pages: int


# Mock data generator
def generate_mock_transactions(count: int = 100) -> List[Dict[str, Any]]:
    """Generate mock transaction data."""
    import random
    
    transactions = []
    for i in range(count):
        fraud_score = random.random()
        transactions.append({
            "transaction_id": f"txn_{i:06d}",
            "user_id": f"user_{random.randint(1, 100):04d}",
            "merchant_id": f"merchant_{random.randint(1, 50):03d}",
            "merchant_name": f"Merchant {random.randint(1, 50)}",
            "amount": round(random.uniform(10, 5000), 2),
            "currency": random.choice(["USD", "EUR", "GBP"]),
            "timestamp": datetime.utcnow() - timedelta(days=random.randint(0, 30)),
            "fraud_score": fraud_score,
            "is_fraud": fraud_score > 0.7,
            "risk_level": (
                RiskLevel.CRITICAL if fraud_score > 0.9
                else RiskLevel.HIGH if fraud_score > 0.7
                else RiskLevel.MEDIUM if fraud_score > 0.4
                else RiskLevel.LOW
            ),
            "status": (
                TransactionStatus.DECLINED if fraud_score > 0.8
                else TransactionStatus.UNDER_REVIEW if fraud_score > 0.6
                else TransactionStatus.APPROVED
            ),
            "ip_address": f"192.168.{random.randint(1, 255)}.{random.randint(1, 255)}",
            "device_id": f"device_{random.randint(1000, 9999)}",
            "location": {
                "country": random.choice(["US", "UK", "CA", "AU"]),
                "city": random.choice(["New York", "London", "Toronto", "Sydney"]),
            },
            "risk_factors": random.sample([
                "unusual_amount",
                "new_merchant",
                "velocity_check",
                "location_mismatch",
                "device_change",
                "time_anomaly"
            ], k=random.randint(0, 3)),
            "reviewed_by": f"analyst_{random.randint(1, 5)}" if fraud_score > 0.6 else None,
            "reviewed_at": datetime.utcnow() - timedelta(hours=random.randint(1, 24)) if fraud_score > 0.6 else None,
            "notes": "Under investigation" if fraud_score > 0.6 else None,
        })
    
    return transactions


# In-memory data store (replace with database in production)
MOCK_TRANSACTIONS = generate_mock_transactions(1000)


@router.get("/", response_model=PaginatedResponse)
async def get_transaction_history(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    sort_by: SortField = Query(SortField.TIMESTAMP, description="Sort field"),
    sort_order: SortOrder = Query(SortOrder.DESC, description="Sort order"),
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    merchant_id: Optional[str] = Query(None, description="Filter by merchant ID"),
    risk_level: Optional[RiskLevel] = Query(None, description="Filter by risk level"),
    status: Optional[TransactionStatus] = Query(None, description="Filter by status"),
    is_fraud: Optional[bool] = Query(None, description="Filter by fraud classification"),
    current_user: User = Depends(get_current_active_user),
    db = Depends(get_database)
):
    """Get paginated transaction history with filtering and sorting.
    
    Args:
        page: Page number (1-indexed)
        page_size: Number of items per page
        sort_by: Field to sort by
        sort_order: Sort order (asc/desc)
        user_id: Filter by user ID
        merchant_id: Filter by merchant ID
        risk_level: Filter by risk level
        status: Filter by transaction status
        is_fraud: Filter by fraud classification
        current_user: Current authenticated user
        db: Database connection
    
    Returns:
        Paginated transaction list
    """
    
    # Filter transactions
    filtered = MOCK_TRANSACTIONS.copy()
    
    if user_id:
        filtered = [t for t in filtered if t["user_id"] == user_id]
    if merchant_id:
        filtered = [t for t in filtered if t["merchant_id"] == merchant_id]
    if risk_level:
        filtered = [t for t in filtered if t["risk_level"] == risk_level]
    if status:
        filtered = [t for t in filtered if t["status"] == status]
    if is_fraud is not None:
        filtered = [t for t in filtered if t["is_fraud"] == is_fraud]
    
    # Sort transactions
    reverse = (sort_order == SortOrder.DESC)
    if sort_by == SortField.TIMESTAMP:
        filtered.sort(key=lambda x: x["timestamp"], reverse=reverse)
    elif sort_by == SortField.AMOUNT:
        filtered.sort(key=lambda x: x["amount"], reverse=reverse)
    elif sort_by == SortField.FRAUD_SCORE:
        filtered.sort(key=lambda x: x["fraud_score"], reverse=reverse)
    elif sort_by == SortField.RISK_LEVEL:
        risk_order = {RiskLevel.LOW: 0, RiskLevel.MEDIUM: 1, RiskLevel.HIGH: 2, RiskLevel.CRITICAL: 3}
        filtered.sort(key=lambda x: risk_order[x["risk_level"]], reverse=reverse)
    
    # Paginate
    total = len(filtered)
    total_pages = (total + page_size - 1) // page_size
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    
    items = [TransactionSummary(**t) for t in filtered[start_idx:end_idx]]
    
    return PaginatedResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages
    )


@router.post("/search", response_model=PaginatedResponse)
async def search_transactions(
    filters: SearchFilters,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    sort_by: SortField = Query(SortField.TIMESTAMP),
    sort_order: SortOrder = Query(SortOrder.DESC),
    current_user: User = Depends(get_current_active_user),
    db = Depends(get_database)
):
    """Advanced transaction search with multiple filters.
    
    Args:
        filters: Search filter parameters
        page: Page number
        page_size: Items per page
        sort_by: Sort field
        sort_order: Sort order
        current_user: Current authenticated user
        db: Database connection
    
    Returns:
        Paginated search results
    """
    
    # Apply filters
    filtered = MOCK_TRANSACTIONS.copy()
    
    if filters.user_id:
        filtered = [t for t in filtered if t["user_id"] == filters.user_id]
    if filters.merchant_id:
        filtered = [t for t in filtered if t["merchant_id"] == filters.merchant_id]
    if filters.min_amount is not None:
        filtered = [t for t in filtered if t["amount"] >= filters.min_amount]
    if filters.max_amount is not None:
        filtered = [t for t in filtered if t["amount"] <= filters.max_amount]
    if filters.start_date:
        filtered = [t for t in filtered if t["timestamp"] >= filters.start_date]
    if filters.end_date:
        filtered = [t for t in filtered if t["timestamp"] <= filters.end_date]
    if filters.risk_levels:
        filtered = [t for t in filtered if t["risk_level"] in filters.risk_levels]
    if filters.statuses:
        filtered = [t for t in filtered if t["status"] in filters.statuses]
    if filters.min_fraud_score is not None:
        filtered = [t for t in filtered if t["fraud_score"] >= filters.min_fraud_score]
    if filters.max_fraud_score is not None:
        filtered = [t for t in filtered if t["fraud_score"] <= filters.max_fraud_score]
    if filters.is_fraud is not None:
        filtered = [t for t in filtered if t["is_fraud"] == filters.is_fraud]
    if filters.search_query:
        query = filters.search_query.lower()
        filtered = [
            t for t in filtered
            if query in t["transaction_id"].lower()
            or query in t["user_id"].lower()
            or query in t["merchant_name"].lower()
        ]
    
    # Sort and paginate (reuse logic from get_transaction_history)
    reverse = (sort_order == SortOrder.DESC)
    if sort_by == SortField.TIMESTAMP:
        filtered.sort(key=lambda x: x["timestamp"], reverse=reverse)
    elif sort_by == SortField.AMOUNT:
        filtered.sort(key=lambda x: x["amount"], reverse=reverse)
    elif sort_by == SortField.FRAUD_SCORE:
        filtered.sort(key=lambda x: x["fraud_score"], reverse=reverse)
    
    total = len(filtered)
    total_pages = (total + page_size - 1) // page_size
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    
    items = [TransactionSummary(**t) for t in filtered[start_idx:end_idx]]
    
    return PaginatedResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages
    )


@router.get("/{transaction_id}", response_model=TransactionDetail)
async def get_transaction_detail(
    transaction_id: str,
    current_user: User = Depends(get_current_active_user),
    db = Depends(get_database)
):
    """Get detailed information for a specific transaction.
    
    Args:
        transaction_id: Transaction identifier
        current_user: Current authenticated user
        db: Database connection
    
    Returns:
        Detailed transaction information
    """
    
    # Find transaction
    transaction = next((t for t in MOCK_TRANSACTIONS if t["transaction_id"] == transaction_id), None)
    
    if not transaction:
        raise HTTPException(status_code=404, detail="Transaction not found")
    
    return TransactionDetail(**transaction)


@router.get("/stats/summary")
async def get_transaction_stats(
    days: int = Query(30, ge=1, le=365, description="Number of days to analyze"),
    current_user: User = Depends(get_current_active_user),
    db = Depends(get_database)
):
    """Get transaction statistics summary.
    
    Args:
        days: Number of days to analyze
        current_user: Current authenticated user
        db: Database connection
    
    Returns:
        Transaction statistics
    """
    
    cutoff_date = datetime.utcnow() - timedelta(days=days)
    recent_transactions = [t for t in MOCK_TRANSACTIONS if t["timestamp"] >= cutoff_date]
    
    total_transactions = len(recent_transactions)
    fraud_transactions = sum(1 for t in recent_transactions if t["is_fraud"])
    total_amount = sum(t["amount"] for t in recent_transactions)
    fraud_amount = sum(t["amount"] for t in recent_transactions if t["is_fraud"])
    
    return {
        "period_days": days,
        "total_transactions": total_transactions,
        "fraud_transactions": fraud_transactions,
        "fraud_rate": fraud_transactions / total_transactions if total_transactions > 0 else 0,
        "total_amount": round(total_amount, 2),
        "fraud_amount": round(fraud_amount, 2),
        "amount_saved": round(fraud_amount, 2),
        "avg_transaction_amount": round(total_amount / total_transactions, 2) if total_transactions > 0 else 0,
        "risk_distribution": {
            "low": sum(1 for t in recent_transactions if t["risk_level"] == RiskLevel.LOW),
            "medium": sum(1 for t in recent_transactions if t["risk_level"] == RiskLevel.MEDIUM),
            "high": sum(1 for t in recent_transactions if t["risk_level"] == RiskLevel.HIGH),
            "critical": sum(1 for t in recent_transactions if t["risk_level"] == RiskLevel.CRITICAL),
        },
    }


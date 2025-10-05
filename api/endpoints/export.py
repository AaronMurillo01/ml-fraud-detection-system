"""Export endpoints for fraud detection data.

This module provides endpoints for exporting:
- Transaction history
- Fraud reports
- Analytics data
- Audit logs

Supported formats: CSV, Excel, JSON, PDF
"""

import logging
import io
import csv
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from enum import Enum

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel, Field
import pandas as pd

from api.auth import get_current_active_user, User, require_role, UserRole
from api.dependencies import get_database

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/export", tags=["Export"])


class ExportFormat(str, Enum):
    """Supported export formats."""
    CSV = "csv"
    EXCEL = "excel"
    JSON = "json"
    PDF = "pdf"


class ExportType(str, Enum):
    """Types of data that can be exported."""
    TRANSACTIONS = "transactions"
    FRAUD_ALERTS = "fraud_alerts"
    ANALYTICS = "analytics"
    AUDIT_LOGS = "audit_logs"
    USER_ACTIVITY = "user_activity"


class ExportRequest(BaseModel):
    """Request model for data export."""
    export_type: ExportType = Field(..., description="Type of data to export")
    format: ExportFormat = Field(ExportFormat.CSV, description="Export format")
    start_date: Optional[datetime] = Field(None, description="Start date for filtering")
    end_date: Optional[datetime] = Field(None, description="End date for filtering")
    filters: Optional[Dict[str, Any]] = Field(None, description="Additional filters")
    include_metadata: bool = Field(True, description="Include metadata in export")


def generate_csv(data: List[Dict[str, Any]], filename: str) -> StreamingResponse:
    """Generate CSV export."""
    output = io.StringIO()
    
    if data:
        writer = csv.DictWriter(output, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)
    
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}.csv"}
    )


def generate_excel(data: List[Dict[str, Any]], filename: str) -> StreamingResponse:
    """Generate Excel export."""
    output = io.BytesIO()
    
    if data:
        df = pd.DataFrame(data)
        df.to_excel(output, index=False, engine='openpyxl')
        output.seek(0)
    
    return StreamingResponse(
        output,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename={filename}.xlsx"}
    )


def generate_json(data: List[Dict[str, Any]], metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Generate JSON export."""
    return {
        "metadata": metadata,
        "data": data,
        "exported_at": datetime.utcnow().isoformat(),
    }


def generate_pdf(data: List[Dict[str, Any]], title: str, filename: str) -> StreamingResponse:
    """Generate PDF export (simplified version)."""
    # Note: In production, use a proper PDF library like ReportLab
    html_content = f"""
    <html>
    <head>
        <title>{title}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #333; }}
            table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #4CAF50; color: white; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <h1>{title}</h1>
        <p>Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
        <p>Total Records: {len(data)}</p>
        <table>
            <tr>
                {''.join(f'<th>{key}</th>' for key in (data[0].keys() if data else []))}
            </tr>
            {''.join(f"<tr>{''.join(f'<td>{value}</td>' for value in row.values())}</tr>" for row in data[:100])}
        </table>
        {f'<p><em>Showing first 100 of {len(data)} records</em></p>' if len(data) > 100 else ''}
    </body>
    </html>
    """
    
    return Response(
        content=html_content,
        media_type="text/html",
        headers={"Content-Disposition": f"attachment; filename={filename}.html"}
    )


async def fetch_transactions(
    start_date: Optional[datetime],
    end_date: Optional[datetime],
    filters: Optional[Dict[str, Any]],
    user: User,
    db
) -> List[Dict[str, Any]]:
    """Fetch transaction data for export."""
    # Mock data - replace with actual database query
    transactions = [
        {
            "transaction_id": f"txn_{i}",
            "user_id": f"user_{i % 100}",
            "amount": 100.0 + (i * 10),
            "merchant": f"Merchant {i % 50}",
            "timestamp": (datetime.utcnow() - timedelta(days=i)).isoformat(),
            "fraud_score": 0.1 + (i % 10) * 0.05,
            "is_fraud": (i % 10) == 0,
            "risk_level": ["low", "medium", "high"][i % 3],
        }
        for i in range(100)
    ]
    
    # Apply filters
    if start_date:
        transactions = [t for t in transactions if datetime.fromisoformat(t["timestamp"]) >= start_date]
    if end_date:
        transactions = [t for t in transactions if datetime.fromisoformat(t["timestamp"]) <= end_date]
    
    return transactions


async def fetch_fraud_alerts(
    start_date: Optional[datetime],
    end_date: Optional[datetime],
    filters: Optional[Dict[str, Any]],
    user: User,
    db
) -> List[Dict[str, Any]]:
    """Fetch fraud alert data for export."""
    # Mock data - replace with actual database query
    alerts = [
        {
            "alert_id": f"alert_{i}",
            "transaction_id": f"txn_{i}",
            "severity": ["low", "medium", "high", "critical"][i % 4],
            "fraud_score": 0.7 + (i % 3) * 0.1,
            "reason": f"Suspicious pattern detected: {['unusual_amount', 'new_merchant', 'velocity'][i % 3]}",
            "timestamp": (datetime.utcnow() - timedelta(hours=i)).isoformat(),
            "status": ["open", "investigating", "resolved", "false_positive"][i % 4],
        }
        for i in range(50)
    ]
    
    return alerts


async def fetch_analytics_data(
    start_date: Optional[datetime],
    end_date: Optional[datetime],
    filters: Optional[Dict[str, Any]],
    user: User,
    db
) -> List[Dict[str, Any]]:
    """Fetch analytics data for export."""
    # Mock data - replace with actual analytics query
    analytics = [
        {
            "date": (datetime.utcnow() - timedelta(days=i)).strftime('%Y-%m-%d'),
            "total_transactions": 1000 + (i * 50),
            "fraud_detected": 10 + (i * 2),
            "fraud_rate": 0.01 + (i * 0.001),
            "avg_transaction_amount": 150.0 + (i * 5),
            "total_amount_blocked": 5000.0 + (i * 100),
        }
        for i in range(30)
    ]
    
    return analytics


async def fetch_audit_logs(
    start_date: Optional[datetime],
    end_date: Optional[datetime],
    filters: Optional[Dict[str, Any]],
    user: User,
    db
) -> List[Dict[str, Any]]:
    """Fetch audit log data for export."""
    # Mock data - replace with actual audit log query
    logs = [
        {
            "log_id": f"log_{i}",
            "user_id": user.id,
            "action": ["login", "logout", "view_transaction", "update_threshold", "export_data"][i % 5],
            "resource": f"resource_{i}",
            "timestamp": (datetime.utcnow() - timedelta(minutes=i * 10)).isoformat(),
            "ip_address": f"192.168.1.{i % 255}",
            "status": "success" if i % 10 != 0 else "failed",
        }
        for i in range(100)
    ]
    
    return logs


@router.post("/")
async def export_data(
    request: ExportRequest,
    current_user: User = Depends(get_current_active_user),
    db = Depends(get_database)
):
    """Export data in specified format.
    
    Args:
        request: Export request parameters
        current_user: Current authenticated user
        db: Database connection
    
    Returns:
        Exported data in requested format
    """
    
    # Fetch data based on export type
    if request.export_type == ExportType.TRANSACTIONS:
        data = await fetch_transactions(
            request.start_date,
            request.end_date,
            request.filters,
            current_user,
            db
        )
        filename = f"transactions_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        title = "Transaction Export"
    
    elif request.export_type == ExportType.FRAUD_ALERTS:
        data = await fetch_fraud_alerts(
            request.start_date,
            request.end_date,
            request.filters,
            current_user,
            db
        )
        filename = f"fraud_alerts_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        title = "Fraud Alerts Export"
    
    elif request.export_type == ExportType.ANALYTICS:
        data = await fetch_analytics_data(
            request.start_date,
            request.end_date,
            request.filters,
            current_user,
            db
        )
        filename = f"analytics_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        title = "Analytics Export"
    
    elif request.export_type == ExportType.AUDIT_LOGS:
        # Require admin role for audit logs
        if current_user.role != UserRole.ADMIN:
            raise HTTPException(status_code=403, detail="Admin access required for audit logs")
        
        data = await fetch_audit_logs(
            request.start_date,
            request.end_date,
            request.filters,
            current_user,
            db
        )
        filename = f"audit_logs_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        title = "Audit Logs Export"
    
    else:
        raise HTTPException(status_code=400, detail="Invalid export type")
    
    if not data:
        raise HTTPException(status_code=404, detail="No data found for export")
    
    # Generate export in requested format
    if request.format == ExportFormat.CSV:
        return generate_csv(data, filename)
    
    elif request.format == ExportFormat.EXCEL:
        return generate_excel(data, filename)
    
    elif request.format == ExportFormat.JSON:
        metadata = {
            "export_type": request.export_type,
            "start_date": request.start_date.isoformat() if request.start_date else None,
            "end_date": request.end_date.isoformat() if request.end_date else None,
            "record_count": len(data),
            "exported_by": current_user.username,
        }
        return generate_json(data, metadata)
    
    elif request.format == ExportFormat.PDF:
        return generate_pdf(data, title, filename)
    
    else:
        raise HTTPException(status_code=400, detail="Invalid export format")


@router.get("/templates")
async def get_export_templates(
    current_user: User = Depends(get_current_active_user)
):
    """Get available export templates and formats.
    
    Returns:
        Available export options
    """
    return {
        "export_types": [
            {
                "type": ExportType.TRANSACTIONS,
                "name": "Transactions",
                "description": "Export transaction history with fraud scores",
                "formats": [ExportFormat.CSV, ExportFormat.EXCEL, ExportFormat.JSON, ExportFormat.PDF],
            },
            {
                "type": ExportType.FRAUD_ALERTS,
                "name": "Fraud Alerts",
                "description": "Export fraud alerts and investigations",
                "formats": [ExportFormat.CSV, ExportFormat.EXCEL, ExportFormat.JSON, ExportFormat.PDF],
            },
            {
                "type": ExportType.ANALYTICS,
                "name": "Analytics",
                "description": "Export analytics and statistics",
                "formats": [ExportFormat.CSV, ExportFormat.EXCEL, ExportFormat.JSON],
            },
            {
                "type": ExportType.AUDIT_LOGS,
                "name": "Audit Logs",
                "description": "Export system audit logs (admin only)",
                "formats": [ExportFormat.CSV, ExportFormat.JSON],
                "requires_admin": True,
            },
        ]
    }


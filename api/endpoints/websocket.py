"""WebSocket endpoints for real-time fraud detection updates.

This module provides WebSocket connections for:
- Real-time transaction monitoring
- Live fraud alerts
- System status updates
- Model performance metrics
"""

import logging
import json
import asyncio
from typing import Dict, Set, Optional, Any
from datetime import datetime

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, Query
from fastapi.websockets import WebSocketState

from api.auth import get_current_user, User
from config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(prefix="/ws", tags=["WebSocket"])


class ConnectionManager:
    """Manages WebSocket connections and broadcasts."""
    
    def __init__(self):
        # Active connections by user ID
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        
        # Connection metadata
        self.connection_metadata: Dict[WebSocket, Dict[str, Any]] = {}
        
        # Subscription channels
        self.subscriptions: Dict[str, Set[WebSocket]] = {
            "transactions": set(),
            "alerts": set(),
            "metrics": set(),
            "system": set(),
        }
    
    async def connect(
        self,
        websocket: WebSocket,
        user_id: str,
        channels: Optional[list] = None
    ):
        """Accept and register a new WebSocket connection."""
        await websocket.accept()
        
        # Add to active connections
        if user_id not in self.active_connections:
            self.active_connections[user_id] = set()
        self.active_connections[user_id].add(websocket)
        
        # Store metadata
        self.connection_metadata[websocket] = {
            "user_id": user_id,
            "connected_at": datetime.utcnow(),
            "channels": channels or ["transactions", "alerts"],
        }
        
        # Subscribe to channels
        for channel in (channels or ["transactions", "alerts"]):
            if channel in self.subscriptions:
                self.subscriptions[channel].add(websocket)
        
        logger.info(
            f"WebSocket connected: user={user_id}, channels={channels}",
            extra={"user_id": user_id}
        )
        
        # Send welcome message
        await self.send_personal_message(
            {
                "type": "connection",
                "status": "connected",
                "message": "Successfully connected to fraud detection system",
                "channels": channels or ["transactions", "alerts"],
                "timestamp": datetime.utcnow().isoformat(),
            },
            websocket
        )
    
    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection."""
        metadata = self.connection_metadata.get(websocket, {})
        user_id = metadata.get("user_id")
        
        # Remove from active connections
        if user_id and user_id in self.active_connections:
            self.active_connections[user_id].discard(websocket)
            if not self.active_connections[user_id]:
                del self.active_connections[user_id]
        
        # Remove from subscriptions
        for channel_connections in self.subscriptions.values():
            channel_connections.discard(websocket)
        
        # Remove metadata
        if websocket in self.connection_metadata:
            del self.connection_metadata[websocket]
        
        logger.info(
            f"WebSocket disconnected: user={user_id}",
            extra={"user_id": user_id}
        )
    
    async def send_personal_message(self, message: dict, websocket: WebSocket):
        """Send a message to a specific connection."""
        try:
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Error sending personal message: {e}")
            self.disconnect(websocket)
    
    async def broadcast_to_user(self, message: dict, user_id: str):
        """Broadcast a message to all connections of a specific user."""
        if user_id in self.active_connections:
            disconnected = set()
            for websocket in self.active_connections[user_id]:
                try:
                    if websocket.client_state == WebSocketState.CONNECTED:
                        await websocket.send_json(message)
                    else:
                        disconnected.add(websocket)
                except Exception as e:
                    logger.error(f"Error broadcasting to user: {e}")
                    disconnected.add(websocket)
            
            # Clean up disconnected websockets
            for ws in disconnected:
                self.disconnect(ws)
    
    async def broadcast_to_channel(self, message: dict, channel: str):
        """Broadcast a message to all subscribers of a channel."""
        if channel not in self.subscriptions:
            logger.warning(f"Unknown channel: {channel}")
            return
        
        disconnected = set()
        for websocket in self.subscriptions[channel]:
            try:
                if websocket.client_state == WebSocketState.CONNECTED:
                    await websocket.send_json(message)
                else:
                    disconnected.add(websocket)
            except Exception as e:
                logger.error(f"Error broadcasting to channel {channel}: {e}")
                disconnected.add(websocket)
        
        # Clean up disconnected websockets
        for ws in disconnected:
            self.disconnect(ws)
    
    async def broadcast_all(self, message: dict):
        """Broadcast a message to all active connections."""
        for user_connections in self.active_connections.values():
            for websocket in user_connections:
                try:
                    if websocket.client_state == WebSocketState.CONNECTED:
                        await websocket.send_json(message)
                except Exception as e:
                    logger.error(f"Error broadcasting to all: {e}")
    
    def get_stats(self) -> dict:
        """Get connection statistics."""
        return {
            "total_connections": sum(len(conns) for conns in self.active_connections.values()),
            "unique_users": len(self.active_connections),
            "channel_subscriptions": {
                channel: len(connections)
                for channel, connections in self.subscriptions.items()
            },
        }


# Global connection manager
manager = ConnectionManager()


@router.websocket("/fraud-updates")
async def websocket_fraud_updates(
    websocket: WebSocket,
    token: Optional[str] = Query(None),
    channels: Optional[str] = Query("transactions,alerts")
):
    """WebSocket endpoint for real-time fraud detection updates.
    
    Args:
        websocket: WebSocket connection
        token: JWT authentication token
        channels: Comma-separated list of channels to subscribe to
    
    Channels:
        - transactions: Real-time transaction updates
        - alerts: Fraud alerts
        - metrics: System metrics
        - system: System status updates
    """
    
    # Parse channels
    channel_list = [c.strip() for c in channels.split(",")] if channels else ["transactions", "alerts"]
    
    # For demo purposes, accept connection without auth
    # In production, validate token and get user
    user_id = "demo_user"  # Replace with actual user from token
    
    try:
        await manager.connect(websocket, user_id, channel_list)
        
        # Keep connection alive and handle incoming messages
        while True:
            try:
                # Receive message from client
                data = await websocket.receive_text()
                message = json.loads(data)
                
                # Handle different message types
                message_type = message.get("type")
                
                if message_type == "ping":
                    # Respond to ping
                    await manager.send_personal_message(
                        {"type": "pong", "timestamp": datetime.utcnow().isoformat()},
                        websocket
                    )
                
                elif message_type == "subscribe":
                    # Subscribe to additional channels
                    new_channels = message.get("channels", [])
                    for channel in new_channels:
                        if channel in manager.subscriptions:
                            manager.subscriptions[channel].add(websocket)
                    
                    await manager.send_personal_message(
                        {
                            "type": "subscription_updated",
                            "channels": new_channels,
                            "timestamp": datetime.utcnow().isoformat(),
                        },
                        websocket
                    )
                
                elif message_type == "unsubscribe":
                    # Unsubscribe from channels
                    remove_channels = message.get("channels", [])
                    for channel in remove_channels:
                        if channel in manager.subscriptions:
                            manager.subscriptions[channel].discard(websocket)
                    
                    await manager.send_personal_message(
                        {
                            "type": "subscription_updated",
                            "removed_channels": remove_channels,
                            "timestamp": datetime.utcnow().isoformat(),
                        },
                        websocket
                    )
                
                else:
                    # Echo unknown messages
                    await manager.send_personal_message(
                        {
                            "type": "echo",
                            "original_message": message,
                            "timestamp": datetime.utcnow().isoformat(),
                        },
                        websocket
                    )
            
            except WebSocketDisconnect:
                break
            except json.JSONDecodeError:
                await manager.send_personal_message(
                    {
                        "type": "error",
                        "message": "Invalid JSON format",
                        "timestamp": datetime.utcnow().isoformat(),
                    },
                    websocket
                )
            except Exception as e:
                logger.error(f"Error handling WebSocket message: {e}")
                await manager.send_personal_message(
                    {
                        "type": "error",
                        "message": "Internal server error",
                        "timestamp": datetime.utcnow().isoformat(),
                    },
                    websocket
                )
    
    except Exception as e:
        logger.error(f"WebSocket connection error: {e}")
    
    finally:
        manager.disconnect(websocket)


@router.get("/stats")
async def get_websocket_stats():
    """Get WebSocket connection statistics."""
    return manager.get_stats()


# Helper functions for broadcasting updates
async def broadcast_transaction_update(transaction_data: dict):
    """Broadcast a transaction update to all subscribers."""
    message = {
        "type": "transaction",
        "data": transaction_data,
        "timestamp": datetime.utcnow().isoformat(),
    }
    await manager.broadcast_to_channel(message, "transactions")


async def broadcast_fraud_alert(alert_data: dict):
    """Broadcast a fraud alert to all subscribers."""
    message = {
        "type": "alert",
        "severity": alert_data.get("severity", "medium"),
        "data": alert_data,
        "timestamp": datetime.utcnow().isoformat(),
    }
    await manager.broadcast_to_channel(message, "alerts")


async def broadcast_system_metric(metric_data: dict):
    """Broadcast a system metric update to all subscribers."""
    message = {
        "type": "metric",
        "data": metric_data,
        "timestamp": datetime.utcnow().isoformat(),
    }
    await manager.broadcast_to_channel(message, "metrics")


async def broadcast_system_status(status_data: dict):
    """Broadcast a system status update to all subscribers."""
    message = {
        "type": "system_status",
        "data": status_data,
        "timestamp": datetime.utcnow().isoformat(),
    }
    await manager.broadcast_to_channel(message, "system")


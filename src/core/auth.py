"""
Authentication utilities for API security.

Provides API key-based authentication for FastAPI endpoints.
Supports both environment variable and MongoDB-based authentication.
"""

from typing import Optional
from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader
from core.config import config
import logging

logger = logging.getLogger(__name__)

# API Key header security scheme
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# MongoDB connection (lazy initialized)
_mongo_client = None
_mongo_db = None


def get_mongo_db():
    """Get MongoDB database connection (lazy initialization)."""
    global _mongo_client, _mongo_db

    if not config.USE_MONGODB:
        return None

    if _mongo_db is None:
        try:
            from pymongo import MongoClient
            _mongo_client = MongoClient(config.MONGODB_URL)
            _mongo_db = _mongo_client[config.MONGODB_DB]
            logger.info(f"Connected to MongoDB: {config.MONGODB_DB}")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            return None

    return _mongo_db


async def verify_api_key_mongodb(api_key: str) -> Optional[dict]:
    """
    Verify API key against MongoDB.

    Args:
        api_key: API key to verify

    Returns:
        dict: API key document if valid, None otherwise
    """
    db = get_mongo_db()
    if db is None:
        return None

    try:
        api_keys_collection = db.api_keys
        key_doc = api_keys_collection.find_one({"key": api_key, "active": True})
        return key_doc
    except Exception as e:
        logger.error(f"MongoDB query error: {e}")
        return None


async def verify_api_key(api_key: Optional[str] = Security(api_key_header)) -> str:
    """
    Verify API key from request header.

    Supports both environment variable and MongoDB-based authentication.

    Args:
        api_key: API key from X-API-Key header

    Returns:
        str: The validated API key

    Raises:
        HTTPException: If API key is invalid or missing (when auth required)
    """
    # If authentication is disabled, allow all requests
    if not config.REQUIRE_AUTH:
        return "auth_disabled"

    # Check if API key is provided
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Provide X-API-Key header.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    # Try MongoDB authentication first
    if config.USE_MONGODB:
        key_doc = await verify_api_key_mongodb(api_key)
        if key_doc:
            logger.info(f"API key verified via MongoDB: {key_doc.get('name', 'Unknown')}")
            return api_key

    # Fallback to environment variable authentication
    if config.API_KEY and api_key == config.API_KEY:
        return api_key

    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Invalid API key",
    )

    return api_key


async def verify_admin_key(api_key: Optional[str] = Security(api_key_header)) -> str:
    """
    Verify admin API key for privileged operations.

    Supports both environment variable and MongoDB-based authentication.

    Args:
        api_key: API key from X-API-Key header

    Returns:
        str: The validated admin API key

    Raises:
        HTTPException: If admin API key is invalid or missing
    """
    # If authentication is disabled, allow all requests
    if not config.REQUIRE_AUTH:
        return "auth_disabled"

    # Check if API key is provided
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing admin API key. Provide X-API-Key header.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    # Try MongoDB authentication first
    if config.USE_MONGODB:
        key_doc = await verify_api_key_mongodb(api_key)
        if key_doc and key_doc.get("is_admin", False):
            logger.info(f"Admin key verified via MongoDB: {key_doc.get('name', 'Unknown')}")
            return api_key

    # Fallback to environment variable authentication
    if not config.ADMIN_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Admin API key not configured on server",
        )

    if api_key == config.ADMIN_API_KEY:
        return api_key

    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Invalid admin API key. Admin access required.",
    )

    return api_key


def get_auth_status() -> dict:
    """
    Get current authentication status.

    Returns:
        dict: Authentication configuration status
    """
    return {
        "authentication_enabled": config.REQUIRE_AUTH,
        "api_key_set": bool(config.API_KEY),
        "admin_key_set": bool(config.ADMIN_API_KEY),
    }

"""
Authentication utilities for API security.

Provides API key-based authentication for FastAPI endpoints.
Supports both environment variable and MongoDB-based authentication.
Supports multiple API keys (comma-separated in environment variables).
"""

from typing import Optional, Set
from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader
from src.core.config import config
from src.core.database import get_mongo_db
import logging

logger = logging.getLogger(__name__)

# API Key header security scheme
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def _parse_api_keys(key_string: Optional[str]) -> Set[str]:
    """
    Parse comma-separated API keys from environment variable.

    Args:
        key_string: Comma-separated list of API keys or single key

    Returns:
        Set of API keys (empty if None)
    """
    if not key_string:
        return set()
    return {key.strip() for key in key_string.split(',') if key.strip()}


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

    # Fallback to environment variable authentication (supports multiple keys)
    valid_keys = _parse_api_keys(config.API_KEY)
    if valid_keys and api_key in valid_keys:
        logger.info(f"API key verified via environment variables")
        return api_key

    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Invalid API key",
    )


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

    # Fallback to environment variable authentication (supports multiple keys)
    valid_admin_keys = _parse_api_keys(config.ADMIN_API_KEY)
    if not valid_admin_keys:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Admin API key not configured on server",
        )

    if api_key in valid_admin_keys:
        logger.info(f"Admin key verified via environment variables")
        return api_key

    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Invalid admin API key. Admin access required.",
    )


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

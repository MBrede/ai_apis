"""
Authentication utilities for API security.

Provides API key-based authentication for FastAPI endpoints.
"""

from typing import Optional
from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader
from core.config import config

# API Key header security scheme
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: Optional[str] = Security(api_key_header)) -> str:
    """
    Verify API key from request header.

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

    # Verify API key
    if api_key != config.API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key",
        )

    return api_key


async def verify_admin_key(api_key: Optional[str] = Security(api_key_header)) -> str:
    """
    Verify admin API key for privileged operations.

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

    # Verify admin API key
    if not config.ADMIN_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Admin API key not configured on server",
        )

    if api_key != config.ADMIN_API_KEY:
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

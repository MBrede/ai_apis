"""
Authentication utilities for API security.

Provides API key-based authentication for FastAPI endpoints.
Supports both environment variable and MongoDB-based authentication.
Supports multiple API keys (comma-separated in environment variables).
"""

import logging
from typing import TYPE_CHECKING, Optional, Set

from src.core.config import config
from src.core.database import get_mongo_db

if TYPE_CHECKING:
    from fastapi import HTTPException, Security, status
    from fastapi.security import APIKeyHeader

logger = logging.getLogger(__name__)

# Lazy import of FastAPI dependencies
_api_key_header = None


def _get_api_key_header():
    """Lazy load APIKeyHeader to avoid requiring FastAPI for non-API uses."""
    global _api_key_header
    if _api_key_header is None:
        try:
            from fastapi.security import APIKeyHeader

            _api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
        except ImportError:
            raise ImportError(
                "FastAPI is required for authentication. "
                "Install it with: pip install 'ai-apis[api-core]'"
            )
    return _api_key_header


def get_api_key_dependency():
    """
    Get the FastAPI Security dependency for API key verification.

    Returns:
        Security dependency that extracts API key from X-API-Key header

    Usage in FastAPI:
        from src.core.auth import get_api_key_dependency
        from fastapi import Depends

        @app.get("/endpoint")
        async def endpoint(api_key: str = Depends(get_api_key_dependency())):
            ...
    """
    from fastapi import Security

    return Security(_get_api_key_header())


# For backwards compatibility - returns the actual APIKeyHeader instance
def get_api_key_header():
    """Get the API key header scheme (lazy loaded)."""
    return _get_api_key_header()


# Module-level __getattr__ for lazy loading api_key_header
def __getattr__(name):
    """Lazy load module-level attributes."""
    if name == "api_key_header":
        return get_api_key_header()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


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
    return {key.strip() for key in key_string.split(",") if key.strip()}


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


async def _verify_api_key_impl(api_key: Optional[str]) -> str:
    """
    Internal implementation of API key verification.

    Args:
        api_key: API key to verify

    Returns:
        str: The validated API key

    Raises:
        HTTPException: If API key is invalid or missing (when auth required)
    """
    from fastapi import HTTPException, status

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


async def verify_api_key(api_key: Optional[str] = None) -> str:
    """
    FastAPI dependency for API key verification.

    This function can be used with Depends() in FastAPI endpoints.
    When used without arguments, it will automatically extract the key from headers.

    Usage:
        from fastapi import Depends
        from src.core.auth import verify_api_key

        @app.get("/endpoint")
        async def endpoint(api_key: str = Depends(verify_api_key)):
            ...

    Args:
        api_key: API key from X-API-Key header (auto-injected by FastAPI)

    Returns:
        str: The validated API key

    Raises:
        HTTPException: If API key is invalid or missing
    """
    # If called from FastAPI with Depends(), api_key should be provided
    # If called directly for testing, we can handle None
    if api_key is None:
        try:
            from fastapi import Security

            # This branch is for when used as a dependency
            return await _verify_api_key_impl(api_key)
        except ImportError:
            # FastAPI not available
            return await _verify_api_key_impl(api_key)
    return await _verify_api_key_impl(api_key)


async def _verify_admin_key_impl(api_key: Optional[str]) -> str:
    """
    Internal implementation of admin API key verification.

    Args:
        api_key: API key to verify

    Returns:
        str: The validated admin API key

    Raises:
        HTTPException: If admin API key is invalid or missing
    """
    from fastapi import HTTPException, status

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


async def verify_admin_key(api_key: Optional[str] = None) -> str:
    """
    FastAPI dependency for admin API key verification.

    This function can be used with Depends() in FastAPI endpoints.

    Usage:
        from fastapi import Depends
        from src.core.auth import verify_admin_key

        @app.post("/admin/endpoint")
        async def admin_endpoint(api_key: str = Depends(verify_admin_key)):
            ...

    Args:
        api_key: Admin API key from X-API-Key header (auto-injected by FastAPI)

    Returns:
        str: The validated admin API key

    Raises:
        HTTPException: If admin API key is invalid or missing
    """
    return await _verify_admin_key_impl(api_key)


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

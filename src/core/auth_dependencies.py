"""FastAPI authentication dependencies.

Kept separate from auth.py so that auth.py can be imported by non-FastAPI
containers (e.g. nextcloud sync) without requiring fastapi to be installed.
"""

from fastapi import Request

from src.core.auth import _verify_admin_key_impl, _verify_api_key_impl


async def verify_api_key(request: Request, api_key: str | None = None) -> str:
    """FastAPI dependency — verifies JWT Bearer token or X-API-Key header."""
    return await _verify_api_key_impl(
        api_key, authorization=request.headers.get("Authorization")
    )


async def verify_admin_key(request: Request, api_key: str | None = None) -> str:
    """FastAPI dependency — verifies admin JWT role or admin X-API-Key header."""
    return await _verify_admin_key_impl(
        api_key, authorization=request.headers.get("Authorization")
    )

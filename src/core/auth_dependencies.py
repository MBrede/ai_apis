"""
FastAPI authentication dependencies.

This module provides FastAPI-compatible authentication dependencies.
Separated from core auth to avoid requiring FastAPI for non-API uses.
"""

from src.core.auth import _verify_admin_key_impl, _verify_api_key_impl, get_api_key_header


def create_api_key_dependency():
    """
    Create a FastAPI dependency for API key verification.

    Returns:
        Async function that can be used with Depends()
    """
    from fastapi import Security

    async def verify_key(api_key: str | None = Security(get_api_key_header())) -> str:
        return await _verify_api_key_impl(api_key)

    return verify_key


def create_admin_key_dependency():
    """
    Create a FastAPI dependency for admin key verification.

    Returns:
        Async function that can be used with Depends()
    """
    from fastapi import Security

    async def verify_admin(api_key: str | None = Security(get_api_key_header())) -> str:
        return await _verify_admin_key_impl(api_key)

    return verify_admin


# Create default instances for backwards compatibility
verify_api_key = create_api_key_dependency()
verify_admin_key = create_admin_key_dependency()

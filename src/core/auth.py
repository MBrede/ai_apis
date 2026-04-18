"""
Authentication utilities for API security.

Priority when Keycloak is configured (KEYCLOAK_URL set):
  1. Authorization: Bearer <jwt>  →  validated against Keycloak JWKS
  2. X-API-Key header / api_key query param  →  existing key-based logic

When KEYCLOAK_URL is not set, only API key authentication is used
(no behaviour change from before).
"""

import logging
import time

from fastapi import Request
from src.core.config import config
from src.core.database import get_mongo_db

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy FastAPI helpers
# ---------------------------------------------------------------------------

_api_key_header = None


def _get_api_key_header():
    """Lazy-load APIKeyHeader to avoid importing FastAPI in non-API contexts."""
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
    """Return the FastAPI Security dependency for API key extraction."""
    from fastapi import Security

    return Security(_get_api_key_header())


def get_api_key_header():
    """Return the APIKeyHeader instance (lazy loaded)."""
    return _get_api_key_header()


def __getattr__(name: str):
    if name == "api_key_header":
        return get_api_key_header()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


# ---------------------------------------------------------------------------
# Keycloak / JWT helpers
# ---------------------------------------------------------------------------

# PyJWKClient caches the JWKS internally; one instance per realm is enough.
_jwks_client = None
_jwks_client_built_at: float = 0
_JWKS_CLIENT_TTL = 3600  # rebuild client (re-fetch JWKS) at most once per hour


def _get_jwks_client():
    """Return a (cached) PyJWKClient for the configured Keycloak realm."""
    global _jwks_client, _jwks_client_built_at

    if not config.KEYCLOAK_URL:
        return None

    now = time.monotonic()
    if _jwks_client is None or now - _jwks_client_built_at > _JWKS_CLIENT_TTL:
        try:
            from jwt import PyJWKClient

            jwks_url = (
                f"{config.KEYCLOAK_URL.rstrip('/')}/realms/{config.KEYCLOAK_REALM}"
                "/protocol/openid-connect/certs"
            )
            _jwks_client = PyJWKClient(
                jwks_url,
                # Honour the KEYCLOAK_VERIFY_SSL flag for self-signed certs
                ssl_context=_build_ssl_context(),
            )
            _jwks_client_built_at = now
            logger.debug("Built PyJWKClient for %s", jwks_url)
        except ImportError:
            raise ImportError(
                "PyJWT[crypto] is required for Keycloak auth. "
                "Install it with: pip install 'PyJWT[crypto]'"
            )

    return _jwks_client


def _build_ssl_context():
    """Return an ssl.SSLContext respecting KEYCLOAK_VERIFY_SSL."""
    import ssl

    if config.KEYCLOAK_VERIFY_SSL:
        return ssl.create_default_context()
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def _validate_jwt(token: str) -> dict:
    """Validate a JWT Bearer token against Keycloak JWKS.

    Args:
        token: Raw JWT string (without 'Bearer ' prefix).

    Returns:
        Decoded JWT payload dict.

    Raises:
        jwt.exceptions.PyJWTError: On any validation failure.
    """
    import jwt as pyjwt

    client = _get_jwks_client()
    signing_key = client.get_signing_key_from_jwt(token)

    decode_opts: dict = {"verify_exp": True}
    audience = config.KEYCLOAK_CLIENT_ID  # None → skip audience check

    payload = pyjwt.decode(
        token,
        signing_key.key,
        algorithms=["RS256"],
        audience=audience,
        options=decode_opts,
    )
    return payload


def _roles_from_payload(payload: dict) -> set[str]:
    """Extract realm-level roles from a decoded JWT payload."""
    return set(payload.get("realm_access", {}).get("roles", []))


# ---------------------------------------------------------------------------
# Token fetching for outbound service calls (Client Credentials)
# ---------------------------------------------------------------------------

_cc_token: str | None = None
_cc_token_expires_at: float = 0


def get_service_token() -> str | None:
    """Obtain a Bearer token for service-to-service calls via Client Credentials.

    Returns None if Keycloak is not configured (callers fall back to API key).
    Token is cached until 30 seconds before expiry.
    """
    global _cc_token, _cc_token_expires_at

    if not config.KEYCLOAK_URL or not config.KEYCLOAK_CLIENT_ID or not config.KEYCLOAK_CLIENT_SECRET:
        return None

    now = time.monotonic()
    if _cc_token and now < _cc_token_expires_at:
        return _cc_token

    import requests as _req

    token_url = (
        f"{config.KEYCLOAK_URL.rstrip('/')}/realms/{config.KEYCLOAK_REALM}"
        "/protocol/openid-connect/token"
    )
    resp = _req.post(
        token_url,
        data={
            "grant_type": "client_credentials",
            "client_id": config.KEYCLOAK_CLIENT_ID,
            "client_secret": config.KEYCLOAK_CLIENT_SECRET,
        },
        verify=config.KEYCLOAK_VERIFY_SSL,
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    _cc_token = data["access_token"]
    _cc_token_expires_at = now + data.get("expires_in", 300) - 30  # 30-s buffer
    logger.debug("Obtained new Keycloak service token (expires in %ds)", data.get("expires_in", 300))
    return _cc_token


def build_auth_headers(api_key: str | None = None) -> dict[str, str]:
    """Return the correct auth headers for outbound API calls.

    When Keycloak is configured the Bearer token is always included.
    If ``api_key`` is also provided it is sent as ``X-API-Key`` alongside
    the Bearer token — this lets admin endpoints fall back to the API key
    check when the service-account token lacks the admin role.

    When Keycloak is not configured only ``X-API-Key`` is sent.

    Args:
        api_key: API key to include (admin or regular). Defaults to
                 ``config.API_KEY`` when not provided.

    Returns:
        Headers dict ready to pass to ``requests`` or ``aiohttp``.
    """
    token = get_service_token()
    key = api_key or config.API_KEY or ""
    if token:
        headers: dict[str, str] = {"Authorization": f"Bearer {token}"}
        if key:
            headers["X-API-Key"] = key
        return headers
    return {"X-API-Key": key}


# ---------------------------------------------------------------------------
# MongoDB key verification
# ---------------------------------------------------------------------------


def _parse_api_keys(key_string: str | None) -> set[str]:
    """Parse comma-separated API keys from an environment variable string."""
    if not key_string:
        return set()
    return {key.strip() for key in key_string.split(",") if key.strip()}


async def verify_api_key_mongodb(api_key: str) -> dict | None:
    """Verify an API key against MongoDB.

    Args:
        api_key: Key to check.

    Returns:
        Key document if valid and active, None otherwise.
    """
    db = get_mongo_db()
    if db is None:
        return None
    try:
        return db.api_keys.find_one({"key": api_key, "active": True})
    except Exception as exc:
        logger.error("MongoDB query error: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Core verification logic
# ---------------------------------------------------------------------------


async def _verify_jwt_or_raise(authorization: str) -> str:
    """Validate a Bearer token, raise HTTPException on failure."""
    from fastapi import HTTPException, status

    token = authorization[len("Bearer "):]
    try:
        payload = _validate_jwt(token)
        sub = payload.get("sub", "unknown")
        logger.info("JWT verified via Keycloak, subject: %s", sub)
        return f"keycloak:{sub}"
    except Exception as exc:
        logger.warning("JWT validation failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def _verify_api_key_impl(api_key: str | None, authorization: str | None = None) -> str:
    """Core auth logic — used by both verify_api_key and verify_admin_key.

    Args:
        api_key: Key from X-API-Key header or query param.
        authorization: Raw Authorization header value.

    Returns:
        Identifier string for the authenticated principal.
    """
    from fastapi import HTTPException, status

    if not config.REQUIRE_AUTH:
        return "auth_disabled"

    # --- Keycloak JWT path ---
    if config.KEYCLOAK_URL and authorization and authorization.startswith("Bearer "):
        return await _verify_jwt_or_raise(authorization)

    # --- API key path ---
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing credentials. Provide X-API-Key header or Authorization: Bearer token.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    if config.USE_MONGODB:
        key_doc = await verify_api_key_mongodb(api_key)
        if key_doc:
            logger.info("API key verified via MongoDB: %s", key_doc.get("name", "Unknown"))
            return api_key

    valid_keys = _parse_api_keys(config.API_KEY)
    if valid_keys and api_key in valid_keys:
        logger.info("API key verified via environment variable")
        return api_key

    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Invalid API key",
    )


async def _verify_admin_key_impl(api_key: str | None, authorization: str | None = None) -> str:
    """Admin-level auth — requires admin role (Keycloak) or admin key (API key)."""
    from fastapi import HTTPException, status

    if not config.REQUIRE_AUTH:
        return "auth_disabled"

    # --- Keycloak JWT path ---
    # If the token is valid and has the admin role → accept.
    # If the token is valid but lacks the role → fall through to API key check
    # so service accounts can still use ADMIN_API_KEY as X-API-Key alongside
    # (or instead of) a Bearer token.
    # If the token is invalid (bad signature, expired) → reject immediately.
    if config.KEYCLOAK_URL and authorization and authorization.startswith("Bearer "):
        token = authorization[len("Bearer "):]
        try:
            payload = _validate_jwt(token)
        except Exception as exc:
            logger.warning("JWT validation failed: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid bearer token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        if config.KEYCLOAK_ADMIN_ROLE in _roles_from_payload(payload):
            logger.info("Admin role verified via Keycloak JWT")
            return f"keycloak:{payload.get('sub')}"
        # Valid token but no admin role — fall through to API key check
        logger.debug(
            "JWT lacks admin role '%s', falling back to API key check", config.KEYCLOAK_ADMIN_ROLE
        )

    # --- Admin API key path ---
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing admin credentials.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    if config.USE_MONGODB:
        key_doc = await verify_api_key_mongodb(api_key)
        if key_doc and key_doc.get("is_admin", False):
            logger.info("Admin key verified via MongoDB: %s", key_doc.get("name"))
            return api_key

    valid_admin_keys = _parse_api_keys(config.ADMIN_API_KEY)
    if not valid_admin_keys:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Admin API key not configured on server",
        )
    if api_key in valid_admin_keys:
        logger.info("Admin key verified via environment variable")
        return api_key

    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Invalid admin API key",
    )


# ---------------------------------------------------------------------------
# FastAPI dependencies
# ---------------------------------------------------------------------------


async def verify_api_key(request: Request, api_key: str | None = None) -> str:
    """FastAPI dependency — verifies JWT Bearer token or API key.

    Usage::

        from fastapi import Depends
        from src.core.auth import verify_api_key

        @app.get("/endpoint")
        async def endpoint(api_key: str = Depends(verify_api_key)):
            ...
    """
    authorization = request.headers.get("Authorization")
    return await _verify_api_key_impl(api_key, authorization=authorization)


async def verify_admin_key(request: Request, api_key: str | None = None) -> str:
    """FastAPI dependency — verifies admin JWT role or admin API key."""
    authorization = request.headers.get("Authorization")
    return await _verify_admin_key_impl(api_key, authorization=authorization)


def get_auth_status() -> dict:
    """Return current authentication configuration status."""
    return {
        "authentication_enabled": config.REQUIRE_AUTH,
        "api_key_set": bool(config.API_KEY),
        "admin_key_set": bool(config.ADMIN_API_KEY),
        "keycloak_enabled": bool(config.KEYCLOAK_URL),
        "keycloak_realm": config.KEYCLOAK_REALM if config.KEYCLOAK_URL else None,
        "keycloak_ssl_verify": config.KEYCLOAK_VERIFY_SSL,
    }

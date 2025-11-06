"""
Tests for the authentication module.

Tests API key verification, admin key verification, and MongoDB integration.
"""

from unittest.mock import Mock, patch

import pytest

# Skip all tests in this module if fastapi isn't available
pytest.importorskip("fastapi")

from fastapi import HTTPException  # noqa: E402

from src.core import auth  # noqa: E402


class TestParseAPIKeys:
    """Test the _parse_api_keys helper function."""

    def test_parse_single_key(self):
        """Test parsing a single API key."""
        result = auth._parse_api_keys("single-key")
        assert result == {"single-key"}

    def test_parse_multiple_keys(self):
        """Test parsing comma-separated API keys."""
        result = auth._parse_api_keys("key1,key2,key3")
        assert result == {"key1", "key2", "key3"}

    def test_parse_keys_with_whitespace(self):
        """Test parsing API keys with whitespace."""
        result = auth._parse_api_keys("key1 , key2 , key3")
        assert result == {"key1", "key2", "key3"}

    def test_parse_empty_string(self):
        """Test parsing empty string."""
        result = auth._parse_api_keys("")
        assert result == set()

    def test_parse_none(self):
        """Test parsing None."""
        result = auth._parse_api_keys(None)
        assert result == set()

    def test_parse_with_empty_segments(self):
        """Test parsing with empty segments (trailing commas, etc)."""
        result = auth._parse_api_keys("key1,,key2,")
        assert result == {"key1", "key2"}


class TestVerifyAPIKeyMongoDB:
    """Test MongoDB-based API key verification."""

    @pytest.mark.asyncio
    async def test_verify_valid_key(self, mock_mongodb):
        """Test verifying a valid API key from MongoDB."""
        with patch("src.core.auth.get_mongo_db", return_value=mock_mongodb):
            result = await auth.verify_api_key_mongodb("test-db-api-key")

            assert result is not None
            assert result["key"] == "test-db-api-key"
            assert result["active"] is True
            mock_mongodb.api_keys.find_one.assert_called_once_with(
                {"key": "test-db-api-key", "active": True}
            )

    @pytest.mark.asyncio
    async def test_verify_invalid_key(self, mock_mongodb):
        """Test verifying an invalid API key."""
        mock_mongodb.api_keys.find_one = Mock(return_value=None)

        with patch("src.core.auth.get_mongo_db", return_value=mock_mongodb):
            result = await auth.verify_api_key_mongodb("invalid-key")

            assert result is None

    @pytest.mark.asyncio
    async def test_verify_when_mongodb_unavailable(self):
        """Test verification when MongoDB is unavailable."""
        with patch("src.core.auth.get_mongo_db", return_value=None):
            result = await auth.verify_api_key_mongodb("any-key")

            assert result is None

    @pytest.mark.asyncio
    async def test_verify_mongodb_error(self, mock_mongodb):
        """Test handling MongoDB query errors."""
        mock_mongodb.api_keys.find_one = Mock(side_effect=Exception("DB Error"))

        with patch("src.core.auth.get_mongo_db", return_value=mock_mongodb):
            result = await auth.verify_api_key_mongodb("test-key")

            assert result is None


class TestVerifyAPIKey:
    """Test API key verification for regular endpoints."""

    @pytest.mark.asyncio
    async def test_verify_when_auth_disabled(self, mock_env_no_auth):
        """Test that verification passes when auth is disabled."""
        with patch("src.core.auth.config.REQUIRE_AUTH", False):
            result = await auth.verify_api_key(api_key=None)
            assert result == "auth_disabled"

    @pytest.mark.asyncio
    async def test_verify_missing_key_when_auth_required(self, mock_env_vars):
        """Test that missing API key raises 401 when auth is required."""
        with patch("src.core.auth.config.REQUIRE_AUTH", True):
            with pytest.raises(HTTPException) as exc_info:
                await auth.verify_api_key(api_key=None)

            assert exc_info.value.status_code == 401
            assert "Missing API key" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_verify_valid_key_from_env(self, mock_env_vars):
        """Test verifying a valid API key from environment variables."""
        with patch("src.core.auth.config.REQUIRE_AUTH", True):
            with patch("src.core.auth.config.USE_MONGODB", False):
                with patch("src.core.auth.config.API_KEY", "test-api-key-1,test-api-key-2"):
                    result = await auth.verify_api_key(api_key="test-api-key-1")
                    assert result == "test-api-key-1"

                    result = await auth.verify_api_key(api_key="test-api-key-2")
                    assert result == "test-api-key-2"

    @pytest.mark.asyncio
    async def test_verify_invalid_key_from_env(self, mock_env_vars):
        """Test that invalid API key raises 403."""
        with patch("src.core.auth.config.REQUIRE_AUTH", True):
            with patch("src.core.auth.config.USE_MONGODB", False):
                with patch("src.core.auth.config.API_KEY", "valid-key"):
                    with pytest.raises(HTTPException) as exc_info:
                        await auth.verify_api_key(api_key="invalid-key")

                    assert exc_info.value.status_code == 403
                    assert "Invalid API key" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_verify_mongodb_takes_precedence(self, mock_mongodb):
        """Test that MongoDB verification is tried before env vars."""
        with patch("src.core.auth.config.REQUIRE_AUTH", True):
            with patch("src.core.auth.config.USE_MONGODB", True):
                with patch("src.core.auth.get_mongo_db", return_value=mock_mongodb):
                    result = await auth.verify_api_key(api_key="test-db-api-key")

                    assert result == "test-db-api-key"
                    mock_mongodb.api_keys.find_one.assert_called_once()

    @pytest.mark.asyncio
    async def test_verify_fallback_to_env_when_mongodb_fails(self, mock_mongodb):
        """Test fallback to env vars when MongoDB doesn't have the key."""
        mock_mongodb.api_keys.find_one = Mock(return_value=None)

        with patch("src.core.auth.config.REQUIRE_AUTH", True):
            with patch("src.core.auth.config.USE_MONGODB", True):
                with patch("src.core.auth.config.API_KEY", "env-key"):
                    with patch("src.core.auth.get_mongo_db", return_value=mock_mongodb):
                        result = await auth.verify_api_key(api_key="env-key")

                        assert result == "env-key"


class TestVerifyAdminKey:
    """Test admin API key verification."""

    @pytest.mark.asyncio
    async def test_verify_admin_when_auth_disabled(self, mock_env_no_auth):
        """Test that admin verification passes when auth is disabled."""
        with patch("src.core.auth.config.REQUIRE_AUTH", False):
            result = await auth.verify_admin_key(api_key=None)
            assert result == "auth_disabled"

    @pytest.mark.asyncio
    async def test_verify_admin_missing_key(self, mock_env_vars):
        """Test that missing admin key raises 401."""
        with patch("src.core.auth.config.REQUIRE_AUTH", True):
            with pytest.raises(HTTPException) as exc_info:
                await auth.verify_admin_key(api_key=None)

            assert exc_info.value.status_code == 401
            assert "Missing admin API key" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_verify_admin_valid_key_from_env(self, mock_env_vars):
        """Test verifying a valid admin key from environment variables."""
        with patch("src.core.auth.config.REQUIRE_AUTH", True):
            with patch("src.core.auth.config.USE_MONGODB", False):
                with patch("src.core.auth.config.ADMIN_API_KEY", "test-admin-key"):
                    result = await auth.verify_admin_key(api_key="test-admin-key")
                    assert result == "test-admin-key"

    @pytest.mark.asyncio
    async def test_verify_admin_invalid_key(self, mock_env_vars):
        """Test that invalid admin key raises 403."""
        with patch("src.core.auth.config.REQUIRE_AUTH", True):
            with patch("src.core.auth.config.USE_MONGODB", False):
                with patch("src.core.auth.config.ADMIN_API_KEY", "valid-admin-key"):
                    with pytest.raises(HTTPException) as exc_info:
                        await auth.verify_admin_key(api_key="invalid-admin-key")

                    assert exc_info.value.status_code == 403
                    assert "Invalid admin API key" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_verify_admin_not_configured(self, mock_env_vars):
        """Test that missing admin key config raises 501."""
        with patch("src.core.auth.config.REQUIRE_AUTH", True):
            with patch("src.core.auth.config.USE_MONGODB", False):
                with patch("src.core.auth.config.ADMIN_API_KEY", None):
                    with pytest.raises(HTTPException) as exc_info:
                        await auth.verify_admin_key(api_key="any-key")

                    assert exc_info.value.status_code == 501
                    assert "not configured" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_verify_admin_from_mongodb(self, mock_mongodb_admin):
        """Test verifying admin key from MongoDB."""
        with patch("src.core.auth.config.REQUIRE_AUTH", True):
            with patch("src.core.auth.config.USE_MONGODB", True):
                with patch("src.core.auth.get_mongo_db", return_value=mock_mongodb_admin):
                    result = await auth.verify_admin_key(api_key="test-admin-db-key")

                    assert result == "test-admin-db-key"
                    mock_mongodb_admin.api_keys.find_one.assert_called_once()

    @pytest.mark.asyncio
    async def test_verify_admin_mongodb_non_admin_key(self, mock_mongodb):
        """Test that non-admin MongoDB key is rejected for admin endpoints."""
        # mock_mongodb has is_admin=False
        with patch("src.core.auth.config.REQUIRE_AUTH", True):
            with patch("src.core.auth.config.USE_MONGODB", True):
                with patch("src.core.auth.config.ADMIN_API_KEY", "fallback-admin-key"):
                    with patch("src.core.auth.get_mongo_db", return_value=mock_mongodb):
                        with pytest.raises(HTTPException) as exc_info:
                            await auth.verify_admin_key(api_key="test-db-api-key")

                        assert exc_info.value.status_code == 403


class TestGetAuthStatus:
    """Test get_auth_status function."""

    def test_get_auth_status_enabled(self, mock_env_vars):
        """Test getting auth status when authentication is enabled."""
        with patch("src.core.auth.config.REQUIRE_AUTH", True):
            with patch("src.core.auth.config.API_KEY", "test-key"):
                with patch("src.core.auth.config.ADMIN_API_KEY", "admin-key"):
                    status = auth.get_auth_status()

                    assert status["authentication_enabled"] is True
                    assert status["api_key_set"] is True
                    assert status["admin_key_set"] is True

    def test_get_auth_status_disabled(self, mock_env_no_auth):
        """Test getting auth status when authentication is disabled."""
        with patch("src.core.auth.config.REQUIRE_AUTH", False):
            with patch("src.core.auth.config.API_KEY", None):
                with patch("src.core.auth.config.ADMIN_API_KEY", None):
                    status = auth.get_auth_status()

                    assert status["authentication_enabled"] is False
                    assert status["api_key_set"] is False
                    assert status["admin_key_set"] is False

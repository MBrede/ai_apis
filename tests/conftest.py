"""
Pytest configuration and fixtures for the AI APIs test suite.

This module provides common fixtures and configuration for all tests.
"""

from unittest.mock import AsyncMock, MagicMock, Mock

import pytest


@pytest.fixture(scope="function")
def mock_env_vars(monkeypatch):
    """
    Mock environment variables for testing.

    Provides a clean environment with test-specific values.
    """
    env_vars = {
        "API_KEY": "test-api-key-1,test-api-key-2",
        "ADMIN_API_KEY": "test-admin-key",
        "HF_TOKEN": "test-hf-token",
        "CIVIT_KEY": "test-civit-key",
        "TELEGRAM_TOKEN": "test-telegram-token",
        "REQUIRE_AUTH": "True",
        "USE_MONGODB": "False",
        "MONGODB_URL": "mongodb://localhost:27017/",
        "MONGODB_DB": "test_ai_apis",
        "DEFAULT_WHISPER_MODEL": "base",
        "DEFAULT_SD_MODEL": "test/model",
        "OLLAMA_HOST": "localhost",
        "OLLAMA_PORT": "2345",
        "SD_HOST": "localhost",
        "SD_PORT": "1234",
        "WHISPER_HOST": "localhost",
        "WHISPER_PORT": "8080",
    }

    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)

    return env_vars


@pytest.fixture(scope="function")
def mock_env_no_auth(monkeypatch):
    """Mock environment with authentication disabled."""
    env_vars = {
        "REQUIRE_AUTH": "False",
        "USE_MONGODB": "False",
    }

    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)

    return env_vars


@pytest.fixture
def mock_mongodb():
    """Mock MongoDB database for testing."""
    mock_db = Mock()
    mock_collection = Mock()

    # Mock api_keys collection
    mock_db.api_keys = mock_collection

    # Mock find_one to return valid key
    mock_collection.find_one = Mock(
        return_value={
            "key": "test-db-api-key",
            "active": True,
            "name": "Test Key",
            "is_admin": False,
        }
    )

    return mock_db


@pytest.fixture
def mock_mongodb_admin():
    """Mock MongoDB database with admin key for testing."""
    mock_db = Mock()
    mock_collection = Mock()

    # Mock api_keys collection
    mock_db.api_keys = mock_collection

    # Mock find_one to return admin key
    mock_collection.find_one = Mock(
        return_value={
            "key": "test-admin-db-key",
            "active": True,
            "name": "Admin Key",
            "is_admin": True,
        }
    )

    return mock_db


@pytest.fixture
def mock_torch():
    """Mock torch for testing without GPU dependencies."""
    mock = MagicMock()
    mock.cuda.is_available = Mock(return_value=True)
    mock.cuda.empty_cache = Mock()
    return mock


@pytest.fixture
def mock_model():
    """Mock ML model for testing."""
    model = Mock()
    model.to = Mock(return_value=model)
    model.__call__ = Mock(return_value="inference_result")
    return model


@pytest.fixture
def mock_pipeline():
    """Mock diffusers pipeline for testing."""
    pipeline = Mock()
    pipeline.to = Mock(return_value=pipeline)
    pipeline.__call__ = Mock(return_value=[Mock()])  # Mock image output
    return pipeline


@pytest.fixture
async def mock_async_client():
    """Mock async HTTP client for testing."""
    client = AsyncMock()
    client.post = AsyncMock(return_value=Mock(status_code=200, json=lambda: {"result": "success"}))
    client.get = AsyncMock(return_value=Mock(status_code=200, json=lambda: {"result": "success"}))
    return client

"""
Tests for the configuration module.

Tests configuration loading, validation, and environment variable handling.
"""

from pathlib import Path

from src.core.config import APIConfig


class TestAPIConfig:
    """Test APIConfig class."""

    def test_config_loads_defaults(self, monkeypatch):
        """Test that config loads with default values when no env vars set."""
        # Clear all relevant env vars
        for key in ["API_KEY", "HF_TOKEN", "TELEGRAM_TOKEN", "USE_MONGODB", "REQUIRE_AUTH"]:
            monkeypatch.delenv(key, raising=False)

        # Create fresh config instance
        config = APIConfig()

        # Check defaults
        assert config.USE_MONGODB is False
        assert config.REQUIRE_AUTH is True
        assert config.MONGODB_DB == "ai_apis"
        assert config.DEFAULT_WHISPER_MODEL == "turbo"
        assert config.REQUEST_TIMEOUT == 300
        assert config.OLLAMA_MODEL == "llama3.3"

    def test_config_loads_from_env(self, mock_env_vars):
        """Test that config correctly loads from environment variables."""
        config = APIConfig()

        assert config.API_KEY == "test-api-key-1,test-api-key-2"
        assert config.ADMIN_API_KEY == "test-admin-key"
        assert config.HF_TOKEN == "test-hf-token"
        assert config.TELEGRAM_TOKEN == "test-telegram-token"
        assert config.REQUIRE_AUTH is True
        assert config.USE_MONGODB is False
        assert config.MONGODB_DB == "test_ai_apis"

    def test_config_boolean_parsing(self, monkeypatch):
        """Test that boolean environment variables are parsed correctly."""
        # Test True values
        monkeypatch.setenv("USE_MONGODB", "true")
        monkeypatch.setenv("REQUIRE_AUTH", "True")
        config = APIConfig()
        assert config.USE_MONGODB is True
        assert config.REQUIRE_AUTH is True

        # Test False values
        monkeypatch.setenv("USE_MONGODB", "false")
        monkeypatch.setenv("REQUIRE_AUTH", "False")
        config = APIConfig()
        assert config.USE_MONGODB is False
        assert config.REQUIRE_AUTH is False

    def test_config_integer_parsing(self, monkeypatch):
        """Test that integer environment variables are parsed correctly."""
        monkeypatch.setenv("SD_PORT", "9999")
        monkeypatch.setenv("REQUEST_TIMEOUT", "600")
        monkeypatch.setenv("OLLAMA_MAX_TOKENS", "4000")

        config = APIConfig()

        assert config.SD_PORT == 9999
        assert config.REQUEST_TIMEOUT == 600
        assert config.OLLAMA_MAX_TOKENS == 4000

    def test_config_float_parsing(self, monkeypatch):
        """Test that float environment variables are parsed correctly."""
        monkeypatch.setenv("OLLAMA_TEMPERATURE", "0.9")

        config = APIConfig()

        assert config.OLLAMA_TEMPERATURE == 0.9

    def test_config_url_construction(self, monkeypatch):
        """Test that URLs are correctly constructed from host and port."""
        monkeypatch.setenv("OLLAMA_HOST", "example.com")
        monkeypatch.setenv("OLLAMA_PORT", "8888")
        monkeypatch.setenv("SD_HOST", "sd.example.com")
        monkeypatch.setenv("SD_PORT", "7777")

        config = APIConfig()

        assert config.OLLAMA_URL == "http://example.com:8888"
        assert config.SD_URL == "http://sd.example.com:7777"

    def test_config_hf_token_variants(self, monkeypatch):
        """Test that HF_TOKEN can be loaded from both lowercase and uppercase env vars."""
        # Test lowercase
        monkeypatch.setenv("hf_token", "lowercase-token")
        monkeypatch.delenv("HF_TOKEN", raising=False)
        config = APIConfig()
        assert config.HF_TOKEN == "lowercase-token"

        # Test uppercase
        monkeypatch.delenv("hf_token", raising=False)
        monkeypatch.setenv("HF_TOKEN", "uppercase-token")
        config = APIConfig()
        assert config.HF_TOKEN == "uppercase-token"

        # Test both (uppercase takes precedence)
        monkeypatch.setenv("hf_token", "lowercase-token")
        monkeypatch.setenv("HF_TOKEN", "uppercase-token")
        config = APIConfig()
        assert config.HF_TOKEN == "uppercase-token"

    def test_config_path_attributes(self):
        """Test that path attributes are correctly set."""
        config = APIConfig()

        assert isinstance(config.PROJECT_ROOT, Path)
        assert isinstance(config.SRC_DIR, Path)
        assert isinstance(config.DATA_DIR, Path)
        assert isinstance(config.USERS_FILE, Path)
        assert isinstance(config.LORA_DIR, Path)

    def test_config_validate_missing_api_key(self, monkeypatch):
        """Test validation warns when API_KEY is missing but auth is required."""
        monkeypatch.setenv("REQUIRE_AUTH", "True")
        monkeypatch.delenv("API_KEY", raising=False)

        config = APIConfig()
        warnings = config.validate()

        assert len(warnings) > 0
        assert any("API_KEY" in warning for warning in warnings)

    def test_config_validate_missing_hf_token(self, monkeypatch):
        """Test validation warns when HF_TOKEN is missing."""
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.delenv("hf_token", raising=False)

        config = APIConfig()
        warnings = config.validate()

        assert any("HF_TOKEN" in warning for warning in warnings)

    def test_config_validate_missing_telegram_token(self, monkeypatch):
        """Test validation warns when TELEGRAM_TOKEN is missing."""
        monkeypatch.delenv("TELEGRAM_TOKEN", raising=False)
        monkeypatch.delenv("telegram_token", raising=False)

        config = APIConfig()
        warnings = config.validate()

        assert any("TELEGRAM_TOKEN" in warning for warning in warnings)

    def test_config_validate_no_warnings_when_complete(self, mock_env_vars):
        """Test that validation returns no warnings when all required config is set."""
        config = APIConfig()
        warnings = config.validate()

        # Should have no warnings when all keys are set
        assert len(warnings) == 0

    def test_config_validate_no_warning_when_auth_disabled(self, monkeypatch):
        """Test that missing API_KEY doesn't warn when auth is disabled."""
        monkeypatch.setenv("REQUIRE_AUTH", "False")
        monkeypatch.delenv("API_KEY", raising=False)
        monkeypatch.setenv("HF_TOKEN", "test-token")
        monkeypatch.setenv("TELEGRAM_TOKEN", "test-token")

        config = APIConfig()
        warnings = config.validate()

        # Should not warn about API_KEY when auth is disabled
        assert not any("API_KEY" in warning for warning in warnings)

    def test_config_print_config(self, mock_env_vars, caplog):
        """Test that print_config logs configuration details."""
        import logging

        caplog.set_level(logging.INFO)

        config = APIConfig()
        config.print_config()

        # Check that important info was logged
        assert any("Configuration" in record.message for record in caplog.records)
        assert any("OLLAMA" in record.message for record in caplog.records)
        assert any("Authentication" in record.message for record in caplog.records)

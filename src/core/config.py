"""
Centralized configuration for all API endpoints and settings.

This module provides configuration management for the SD API project,
including endpoint URLs, authentication settings, and API keys.
Load settings from environment variables when available.
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logger
logger = logging.getLogger(__name__)


class APIConfig:
    """Configuration for API endpoints and authentication."""

    # =============================================================================
    # API Endpoints
    # =============================================================================

    # LLM Endpoints (Legacy - for backward compatibility)
    LLM_MIXTRAL_HOST: str = os.getenv("LLM_MIXTRAL_HOST", "localhost")
    LLM_MIXTRAL_PORT: int = int(os.getenv("LLM_MIXTRAL_PORT", "420420"))
    LLM_MIXTRAL_URL: str = f"http://{LLM_MIXTRAL_HOST}:{LLM_MIXTRAL_PORT}"

    LLM_COMMAND_R_HOST: str = os.getenv("LLM_COMMAND_R_HOST", "localhost")
    LLM_COMMAND_R_PORT: int = int(os.getenv("LLM_COMMAND_R_PORT", "420421"))
    LLM_COMMAND_R_URL: str = f"http://{LLM_COMMAND_R_HOST}:{LLM_COMMAND_R_PORT}"

    # OLLAMA Endpoint (Primary LLM)
    OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "localhost")
    OLLAMA_PORT: int = int(os.getenv("OLLAMA_PORT", "2345"))
    OLLAMA_URL: str = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}"

    # Stable Diffusion Endpoint
    SD_HOST: str = os.getenv("SD_HOST", "localhost")
    SD_PORT: int = int(os.getenv("SD_PORT", "1234"))
    SD_URL: str = f"http://{SD_HOST}:{SD_PORT}"

    # Whisper Endpoint
    WHISPER_HOST: str = os.getenv("WHISPER_HOST", "localhost")
    WHISPER_PORT: int = int(os.getenv("WHISPER_PORT", "8080"))
    WHISPER_URL: str = f"http://{WHISPER_HOST}:{WHISPER_PORT}"

    # =============================================================================
    # Authentication
    # =============================================================================

    # API Key for authentication (IMPORTANT: Set in .env file!)
    API_KEY: str | None = os.getenv("API_KEY")

    # Require authentication (set to False for development only)
    REQUIRE_AUTH: bool = os.getenv("REQUIRE_AUTH", "True").lower() == "true"

    # Admin API key for privileged operations
    ADMIN_API_KEY: str | None = os.getenv("ADMIN_API_KEY")

    # =============================================================================
    # MongoDB Configuration
    # =============================================================================

    # Use MongoDB for authentication and bot settings
    USE_MONGODB: bool = os.getenv("USE_MONGODB", "False").lower() == "true"

    # MongoDB connection URL
    MONGODB_URL: str = os.getenv("MONGODB_URL", "mongodb://localhost:27017/")

    # MongoDB database name
    MONGODB_DB: str = os.getenv("MONGODB_DB", "ai_apis")

    # =============================================================================
    # External Service Tokens
    # =============================================================================

    # HuggingFace token
    HF_TOKEN: str | None = os.getenv("hf_token") or os.getenv("HF_TOKEN")

    # Civitai API key (for LORA downloads)
    CIVIT_KEY: str | None = os.getenv("civit_key") or os.getenv("CIVIT_KEY")

    # Telegram bot token
    TELEGRAM_TOKEN: str | None = os.getenv("telegram_token") or os.getenv("TELEGRAM_TOKEN")

    # =============================================================================
    # File Paths
    # =============================================================================

    PROJECT_ROOT: Path = Path(__file__).parent.parent
    SRC_DIR: Path = PROJECT_ROOT / "src"
    DATA_DIR: Path = PROJECT_ROOT / "data"

    # User data files
    USERS_FILE: Path = Path("users.json")
    CONTACTS_FILE: Path = Path("contacts.json")

    # LORA configuration
    LORA_LIST_FILE: Path = Path("lora_list.json")
    LORA_DIR: Path = Path("loras")

    # LLM endpoints configuration
    LLM_ENDPOINTS_FILE: Path = Path("available_endpoints.json")

    # =============================================================================
    # Model Settings
    # =============================================================================

    # Default Whisper model
    DEFAULT_WHISPER_MODEL: str = os.getenv("DEFAULT_WHISPER_MODEL", "turbo")

    # Default Stable Diffusion model
    DEFAULT_SD_MODEL: str = os.getenv("DEFAULT_SD_MODEL", "stabilityai/stable-diffusion-2-1")

    # Torch dtype
    DEFAULT_TORCH_DTYPE: str = os.getenv("DEFAULT_TORCH_DTYPE", "float16")

    # =============================================================================
    # Server Settings
    # =============================================================================

    # Request timeout (seconds)
    REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT", "300"))

    # Max file upload size (bytes)
    MAX_UPLOAD_SIZE: int = int(os.getenv("MAX_UPLOAD_SIZE", str(100 * 1024 * 1024)))  # 100MB

    # =============================================================================
    # OLLAMA Settings
    # =============================================================================

    # Default OLLAMA model for chat
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama4_long:latest")

    # OLLAMA temperature
    OLLAMA_TEMPERATURE: float = float(os.getenv("OLLAMA_TEMPERATURE", "0.2"))

    # OLLAMA max tokens
    OLLAMA_MAX_TOKENS: int = int(os.getenv("OLLAMA_MAX_TOKENS", "5000"))

    @classmethod
    def validate(cls) -> list[str]:
        """
        Validate configuration and return list of warnings.

        Returns:
            List of warning messages for missing or invalid configuration
        """
        warnings = []

        if cls.REQUIRE_AUTH and not cls.API_KEY:
            warnings.append(
                "⚠️  API_KEY not set but REQUIRE_AUTH=True. "
                "Please set API_KEY in .env file for security!"
            )

        if not cls.HF_TOKEN:
            warnings.append("⚠️  HF_TOKEN not set. HuggingFace model downloads may fail.")

        if not cls.TELEGRAM_TOKEN:
            warnings.append("⚠️  TELEGRAM_TOKEN not set. Telegram bot will not work.")

        return warnings

    @classmethod
    def print_config(cls) -> None:
        """Log current configuration (excluding sensitive data)."""
        logger.info("=" * 70)
        logger.info("SD API Configuration")
        logger.info("=" * 70)
        logger.info(f"OLLAMA URL: {cls.OLLAMA_URL} (model: {cls.OLLAMA_MODEL})")
        logger.info(f"SD URL: {cls.SD_URL}")
        logger.info(f"Whisper URL: {cls.WHISPER_URL}")
        logger.info(f"Authentication: {'Enabled' if cls.REQUIRE_AUTH else 'Disabled'}")
        logger.info(f"API Key: {'✓ Set' if cls.API_KEY else '✗ Not Set'}")
        logger.info(f"HF Token: {'✓ Set' if cls.HF_TOKEN else '✗ Not Set'}")
        logger.info(f"Telegram Token: {'✓ Set' if cls.TELEGRAM_TOKEN else '✗ Not Set'}")
        logger.info(f"MongoDB: {'Enabled' if cls.USE_MONGODB else 'Disabled'}")
        if cls.USE_MONGODB:
            logger.info(f"MongoDB URL: {cls.MONGODB_URL}")
            logger.info(f"MongoDB DB: {cls.MONGODB_DB}")
        logger.info("=" * 70)

        warnings = cls.validate()
        if warnings:
            logger.warning("Configuration Warnings:")
            for warning in warnings:
                logger.warning(f"  {warning}")


# Create singleton instance
config = APIConfig()


# Validate on import (only show warnings if running as main module)
if __name__ == "__main__":
    config.print_config()

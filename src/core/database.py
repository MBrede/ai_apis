"""
Shared MongoDB database connection utilities.

Provides centralized MongoDB connection management with lazy initialization.
Used by authentication system and bot for persistent storage.
"""

import logging
from typing import Optional
from pymongo.database import Database

logger = logging.getLogger(__name__)

# MongoDB connection (lazy initialized)
_mongo_client = None
_mongo_db = None


def get_mongo_db() -> Optional[Database]:
    """
    Get MongoDB database connection with lazy initialization.

    Returns:
        Database: MongoDB database instance if USE_MONGODB=true, None otherwise

    Note:
        Connection is cached globally and reused across calls.
        First call creates the connection, subsequent calls return cached instance.
    """
    global _mongo_client, _mongo_db

    # Import config here to avoid circular imports
    from core.config import config

    if not config.USE_MONGODB:
        return None

    if _mongo_db is None:
        try:
            from pymongo import MongoClient
            _mongo_client = MongoClient(config.MONGODB_URL)
            _mongo_db = _mongo_client[config.MONGODB_DB]
            logger.info(f"Connected to MongoDB: {config.MONGODB_DB}")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            return None

    return _mongo_db


def close_mongo_connection():
    """
    Close MongoDB connection and reset global state.

    Useful for cleanup in tests or graceful shutdown.
    """
    global _mongo_client, _mongo_db

    if _mongo_client is not None:
        try:
            _mongo_client.close()
            logger.info("MongoDB connection closed")
        except Exception as e:
            logger.error(f"Error closing MongoDB connection: {e}")
        finally:
            _mongo_client = None
            _mongo_db = None

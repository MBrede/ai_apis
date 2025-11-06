#!/usr/bin/env python3
"""
MongoDB initialization script for AI APIs.

Creates initial database structure and adds default API keys.
Run this after starting MongoDB for the first time.

Usage:
    python scripts/init_mongodb.py
"""

import sys
import os
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()


def init_mongodb():
    """Initialize MongoDB with collections and indexes."""

    # Get MongoDB connection details
    mongodb_url = os.getenv("MONGODB_URL", "mongodb://localhost:27017/")
    mongodb_db = os.getenv("MONGODB_DB", "ai_apis")
    api_key = os.getenv("API_KEY", "your-api-key-here")
    admin_api_key = os.getenv("ADMIN_API_KEY", "your-admin-key-here")

    print(f"Connecting to MongoDB at {mongodb_url}...")

    try:
        # Connect to MongoDB
        client = MongoClient(mongodb_url)
        db = client[mongodb_db]

        # Test connection
        client.admin.command('ping')
        print(f"✓ Connected to MongoDB database: {mongodb_db}")

        # ==============================================================================
        # Create API Keys Collection
        # ==============================================================================

        api_keys_collection = db.api_keys

        # Create indexes
        api_keys_collection.create_index("key", unique=True)
        api_keys_collection.create_index("active")
        print("✓ Created api_keys collection with indexes")

        # Insert default API keys
        default_keys = [
            {
                "key": api_key,
                "name": "Default API Key",
                "is_admin": False,
                "active": True,
                "created_at": datetime.utcnow(),
                "rate_limit": 1000,  # requests per day
                "usage_count": 0
            },
            {
                "key": admin_api_key,
                "name": "Admin API Key",
                "is_admin": True,
                "active": True,
                "created_at": datetime.utcnow(),
                "rate_limit": 10000,  # requests per day
                "usage_count": 0
            }
        ]

        for key_doc in default_keys:
            try:
                api_keys_collection.insert_one(key_doc)
                print(f"✓ Added API key: {key_doc['name']}")
            except Exception as e:
                if "duplicate key error" in str(e):
                    print(f"! API key already exists: {key_doc['name']}")
                else:
                    print(f"✗ Error adding API key: {e}")

        # ==============================================================================
        # Create Bot Settings Collection
        # ==============================================================================

        bot_settings_collection = db.bot_settings

        # Create indexes
        bot_settings_collection.create_index("user_id", unique=True)
        print("✓ Created bot_settings collection with indexes")

        # Insert default bot settings
        default_bot_settings = {
            "user_id": "default",
            "sd_parameters": {
                "model": "stabilityai/stable-diffusion-2-1",
                "steps": 30,
                "guidance_scale": 7.5,
                "width": 512,
                "height": 512,
                "negative_prompt": "blurry, low quality, distorted"
            },
            "llm_mode": False,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }

        try:
            bot_settings_collection.insert_one(default_bot_settings)
            print("✓ Added default bot settings")
        except Exception as e:
            if "duplicate key error" in str(e):
                print("! Default bot settings already exist")
            else:
                print(f"✗ Error adding bot settings: {e}")

        # ==============================================================================
        # Create Usage Logs Collection
        # ==============================================================================

        usage_logs_collection = db.usage_logs

        # Create indexes for efficient querying
        usage_logs_collection.create_index([("timestamp", -1)])
        usage_logs_collection.create_index([("api_key", 1)])
        usage_logs_collection.create_index([("endpoint", 1)])
        print("✓ Created usage_logs collection with indexes")

        # ==============================================================================
        # Summary
        # ==============================================================================

        print("\n" + "=" * 70)
        print("MongoDB Initialization Complete!")
        print("=" * 70)
        print(f"Database: {mongodb_db}")
        print(f"Collections:")
        print(f"  - api_keys: {api_keys_collection.count_documents({})} documents")
        print(f"  - bot_settings: {bot_settings_collection.count_documents({})} documents")
        print(f"  - usage_logs: {usage_logs_collection.count_documents({})} documents")
        print("\n" + "⚠️  IMPORTANT: Change default API keys in production!")
        print("=" * 70)

        client.close()

    except Exception as e:
        print(f"\n✗ Error initializing MongoDB: {e}")
        sys.exit(1)


if __name__ == "__main__":
    init_mongodb()

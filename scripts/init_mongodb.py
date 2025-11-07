#!/usr/bin/env python3
"""
MongoDB initialization script for AI APIs.

Creates initial database structure, adds default API keys, and manages admin users.
Run this after starting MongoDB for the first time.

Usage:
    python scripts/init_mongodb.py setup [--add-user USER_ID] [--username NAME]
    python scripts/init_mongodb.py add-user USER_ID [--username NAME]
    python scripts/init_mongodb.py setup --add-user USER_ID --username NAME

Examples:
    # Setup database and add admin user
    python scripts/init_mongodb.py setup --add-user 123456789 --username "admin"

    # Add admin user to existing database
    python scripts/init_mongodb.py add-user 123456789 --username "admin"

    # Setup with user from environment variable
    ADMIN_TELEGRAM_USER_ID=123456789 python scripts/init_mongodb.py setup
"""

import argparse
import sys
import os
from datetime import datetime
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()


def get_mongo_connection():
    """
    Get MongoDB connection details from environment variables.

    Returns:
        tuple: (mongodb_url, mongodb_db, mongodb_admin, mongodb_pw)
    """
    mongodb_admin = os.getenv("MONGO_ROOT_USER", "admin")
    mongodb_pw = os.getenv("MONGO_ROOT_PASSWORD", "password")
    mongodb_url = os.getenv("MONGODB_URL", "mongodb://localhost:27017/")
    mongodb_db = os.getenv("MONGODB_DB", "ai_apis")

    return mongodb_url, mongodb_db, mongodb_admin, mongodb_pw


def add_telegram_user(user_id: int, username: str | None = None, is_admin: bool = True) -> bool:
    """
    Add a Telegram user to MongoDB bot_users collection.

    Args:
        user_id: Telegram user ID
        username: Optional username for the user
        is_admin: Whether the user should have admin privileges (default: True)

    Returns:
        bool: True if successful, False otherwise
    """
    mongodb_url, mongodb_db, mongodb_admin, mongodb_pw = get_mongo_connection()

    print(f"\nAdding Telegram user {user_id}...")

    try:
        # Connect to MongoDB
        client = MongoClient(mongodb_url, username=mongodb_admin, password=mongodb_pw)
        db = client[mongodb_db]

        # Test connection
        client.admin.command("ping")
        print(f"✓ Connected to MongoDB database: {mongodb_db}")

        # Get bot_users collection
        bot_users_collection = db.bot_users

        # Create user document
        user_doc = {
            "user_id": user_id,
            "admin": is_admin,
            "mode": "sd",  # Default mode
            "current_settings": {},
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
        }

        if username:
            user_doc["username"] = username

        # Insert or update user
        result = bot_users_collection.update_one(
            {"user_id": user_id},
            {"$set": user_doc, "$setOnInsert": {"created_at": datetime.utcnow()}},
            upsert=True,
        )

        if result.upserted_id:
            print(f"✓ Added new {'admin' if is_admin else 'regular'} user: {user_id}")
            if username:
                print(f"  Username: {username}")
        else:
            print(f"✓ Updated existing user: {user_id}")

        client.close()
        return True

    except Exception as e:
        print(f"\n✗ Error adding user to MongoDB: {e}")
        return False


def init_mongodb(add_user_id: int | None = None, username: str | None = None):
    """
    Initialize MongoDB with collections and indexes.

    Args:
        add_user_id: Optional Telegram user ID to add as admin during setup
        username: Optional username for the admin user
    """
    # Get MongoDB connection details
    mongodb_url, mongodb_db, mongodb_admin, mongodb_pw = get_mongo_connection()
    api_key = os.getenv("API_KEY", "your-api-key-here")
    admin_api_key = os.getenv("ADMIN_API_KEY", "your-admin-key-here")

    print(f"Connecting to MongoDB at {mongodb_url}...")

    try:
        # Connect to MongoDB
        client = MongoClient(mongodb_url, username=mongodb_admin, password=mongodb_pw)
        db = client[mongodb_db]

        # Test connection
        client.admin.command("ping")
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
        # Create Bot Users Collection
        # ==============================================================================

        bot_users_collection = db.bot_users

        # Create indexes
        bot_users_collection.create_index("user_id", unique=True)
        print("✓ Created bot_users collection with indexes")

        # Add admin user if provided
        if add_user_id:
            user_doc = {
                "user_id": add_user_id,
                "admin": True,
                "mode": "sd",
                "current_settings": {},
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
            }

            if username:
                user_doc["username"] = username

            try:
                result = bot_users_collection.update_one(
                    {"user_id": add_user_id},
                    {"$set": user_doc, "$setOnInsert": {"created_at": datetime.utcnow()}},
                    upsert=True,
                )
                if result.upserted_id:
                    print(f"✓ Added admin user: {add_user_id}" + (f" ({username})" if username else ""))
                else:
                    print(f"✓ Updated admin user: {add_user_id}" + (f" ({username})" if username else ""))
            except Exception as e:
                print(f"✗ Error adding admin user: {e}")

        # ==============================================================================
        # Create Bot Contacts Collection
        # ==============================================================================

        bot_contacts_collection = db.bot_contacts

        # Create indexes
        bot_contacts_collection.create_index("user_id", unique=True)
        bot_contacts_collection.create_index([("last_attempt", -1)])
        print("✓ Created bot_contacts collection with indexes")

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

        # Check if at least one admin user exists
        admin_count = bot_users_collection.count_documents({"admin": True})

        print("\n" + "=" * 70)
        print("MongoDB Initialization Complete!")
        print("=" * 70)
        print(f"Database: {mongodb_db}")
        print(f"Collections:")
        print(f"  - api_keys: {api_keys_collection.count_documents({})} documents")
        print(f"  - bot_users: {bot_users_collection.count_documents({})} documents")
        print(f"    - Admin users: {admin_count}")
        print(f"  - bot_contacts: {bot_contacts_collection.count_documents({})} documents")
        print(f"  - usage_logs: {usage_logs_collection.count_documents({})} documents")

        print("\n" + "⚠️  IMPORTANT:")
        print("  - Change default API keys in production!")

        if admin_count == 0:
            print("\n" + "❌ WARNING: No admin users configured!")
            print("   The Telegram bot will not be accessible until you add an admin user.")
            print("   Use one of these methods:")
            print("   1. Run: python scripts/init_mongodb.py add-user <USER_ID> --username <NAME>")
            print("   2. Set ADMIN_TELEGRAM_USER_ID environment variable and run setup again")
            print("   3. Use /add_user command in Telegram (requires existing admin)")
        else:
            print("  ✓ Admin users configured successfully")

        print("=" * 70)

        client.close()

    except Exception as e:
        print(f"\n✗ Error initializing MongoDB: {e}")
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Initialize MongoDB and manage Telegram admin users for AI APIs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Setup database with admin user
  %(prog)s setup --add-user 123456789 --username "admin"

  # Setup database with admin user from environment
  ADMIN_TELEGRAM_USER_ID=123456789 %(prog)s setup

  # Add admin user to existing database
  %(prog)s add-user 123456789 --username "admin"

  # Just setup database (no admin user)
  %(prog)s setup
        """,
    )

    # Create subcommands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Setup command
    setup_parser = subparsers.add_parser(
        "setup", help="Initialize MongoDB collections and indexes"
    )
    setup_parser.add_argument(
        "--add-user",
        type=int,
        metavar="USER_ID",
        help="Telegram user ID to add as admin during setup",
    )
    setup_parser.add_argument(
        "--username", type=str, metavar="NAME", help="Username for the admin user"
    )

    # Add-user command
    add_user_parser = subparsers.add_parser(
        "add-user", help="Add a Telegram admin user to the database"
    )
    add_user_parser.add_argument("user_id", type=int, help="Telegram user ID")
    add_user_parser.add_argument(
        "--username", type=str, metavar="NAME", help="Username for the admin user"
    )

    args = parser.parse_args()

    # Default to 'setup' command if none specified (backward compatibility)
    if not args.command:
        print("No command specified. Use 'setup' or 'add-user'. Running 'setup' for backward compatibility.\n")
        args.command = "setup"
        args.add_user = None
        args.username = None

    # Check for environment variable for admin user
    env_user_id = os.getenv("ADMIN_TELEGRAM_USER_ID")
    env_username = os.getenv("ADMIN_TELEGRAM_USERNAME")

    if args.command == "setup":
        # Determine user_id and username
        user_id = args.add_user if hasattr(args, "add_user") else None
        username = args.username if hasattr(args, "username") else None

        # Use environment variables if CLI args not provided
        if not user_id and env_user_id:
            try:
                user_id = int(env_user_id)
                print(f"Using admin user ID from environment: {user_id}")
            except ValueError:
                print(f"Warning: Invalid ADMIN_TELEGRAM_USER_ID in environment: {env_user_id}")

        if not username and env_username:
            username = env_username
            print(f"Using username from environment: {username}")

        init_mongodb(add_user_id=user_id, username=username)

    elif args.command == "add-user":
        user_id = args.user_id
        username = args.username

        # Add username from environment if not provided
        if not username and env_username:
            username = env_username
            print(f"Using username from environment: {username}")

        success = add_telegram_user(user_id, username)
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

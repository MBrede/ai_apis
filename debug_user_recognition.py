#!/usr/bin/env python3
"""
Diagnostic script to debug user recognition issue.
This script checks the MongoDB bot_users collection and verifies data types.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.database import get_mongo_db


def main():
    print("=" * 70)
    print("USER RECOGNITION DIAGNOSTIC TOOL")
    print("=" * 70)

    # Connect to MongoDB
    db = get_mongo_db()
    if db is None:
        print("❌ ERROR: Could not connect to MongoDB")
        return

    print("✅ Connected to MongoDB successfully\n")

    # Get bot_users collection
    users_collection = db.bot_users

    # Count total users
    total_users = users_collection.count_documents({})
    print(f"📊 Total users in database: {total_users}\n")

    if total_users == 0:
        print("❌ WARNING: No users found in bot_users collection!")
        return

    # Display all users
    print("-" * 70)
    print("USER LIST:")
    print("-" * 70)

    for idx, user_doc in enumerate(users_collection.find(), 1):
        user_id = user_doc.get("user_id")
        username = user_doc.get("username", "N/A")
        admin = user_doc.get("admin", False)
        mode = user_doc.get("mode", "N/A")

        print(f"\n{idx}. User Details:")
        print(f"   user_id: {user_id}")
        print(f"   type: {type(user_id).__name__}")
        print(f"   username: {username}")
        print(f"   admin: {admin}")
        print(f"   mode: {mode}")

        # Show the type mismatch issue
        print(f"\n   🔍 Type Mismatch Check:")
        print(f"      - MongoDB user_id: {user_id} (type: {type(user_id).__name__})")
        print(f"      - As string: '{str(user_id)}' (type: {type(str(user_id)).__name__})")
        print(f"      - Comparison: {user_id} == '{str(user_id)}' → {user_id == str(user_id)}")

        # Simulate the lookup issue
        test_dict_int = {user_id: "value"}  # Dictionary with int key
        test_str = str(user_id)  # String version

        print(f"\n   🧪 Simulating bot.py lookup:")
        print(f"      - Dict with int key: {test_dict_int}")
        print(f"      - Looking up string '{test_str}': {test_str in test_dict_int}")
        print(f"      - Looking up int {user_id}: {user_id in test_dict_int}")

        if test_str not in test_dict_int:
            print(f"      ❌ THIS IS THE BUG! String lookup fails on int keys!")

    print("\n" + "=" * 70)
    print("DIAGNOSIS COMPLETE")
    print("=" * 70)
    print("\n💡 ISSUE IDENTIFIED:")
    print("   - MongoDB stores user_id as INTEGER")
    print("   - bot.py line 44 loads user_id as INTEGER")
    print("   - bot.py line 197 converts Telegram user_id to STRING")
    print("   - bot.py line 198 compares STRING to dict with INT keys")
    print("   - Result: User lookup always fails!\n")
    print("🔧 SOLUTION:")
    print("   Convert user_id to string when loading from MongoDB (line 44)")
    print("   Change: user_id = user_doc['user_id']")
    print("   To:     user_id = str(user_doc['user_id'])")
    print("=" * 70)


if __name__ == "__main__":
    main()

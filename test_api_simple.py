#!/usr/bin/env python3
"""
Simple test script to diagnose API startup issues.
Run directly without gunicorn to see all errors.
"""
import sys
import logging

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

print("=== Starting API import test ===")
print(f"Python: {sys.version}")

try:
    print("\n1. Testing stable_diffusion_api import...")
    from src.image_generation.stable_diffusion_api import app as sd_app
    print("   ✓ stable_diffusion_api imported successfully")

    print("\n2. Testing health endpoint...")
    from fastapi.testclient import TestClient
    client = TestClient(sd_app)

    print("   Calling GET /health...")
    response = client.get("/health")
    print(f"   Status code: {response.status_code}")
    print(f"   Response: {response.json()}")

    if response.status_code == 200:
        print("   ✓ Health check works!")
    else:
        print(f"   ✗ Health check failed with {response.status_code}")

except Exception as e:
    print(f"\n✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n=== All tests passed ===")

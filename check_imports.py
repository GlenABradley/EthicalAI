import sys

def check_import(module_name):
    try:
        __import__(module_name)
        print(f"✓ Successfully imported {module_name}")
        return True
    except ImportError as e:
        print(f"✗ Failed to import {module_name}: {e}")
        return False

print(f"Python {sys.version}")
print("Checking imports...\n")

# Check core Python modules
check_import("os")
check_import("sys")

# Check test dependencies
check_import("pytest")
check_import("fastapi")
check_import("uvicorn")
check_import("httpx")
check_import("pytest_asyncio")
check_import("pytest_cov")

# Check project modules
try:
    check_import("coherence.api.main")
    check_import("coherence.api.axis_registry")
except Exception as e:
    print(f"\nError checking project modules: {e}")
    print("Make sure to run this script from the project root directory.")

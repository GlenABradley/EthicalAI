#!/usr/bin/env python3
"""Direct test runner to bypass command execution issues."""

import subprocess
import sys
import os
from pathlib import Path

def run_test():
    """Run the health check test directly."""
    # Set working directory
    os.chdir(Path(__file__).parent)
    
    # Set up environment
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd() / "src")
    
    # Run pytest with specific test
    cmd = [
        sys.executable, "-m", "pytest", 
        "tests/test_end_to_end.py::test_health_check",
        "-v", "-s", "--tb=short"
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    print(f"Working directory: {Path.cwd()}")
    print(f"Python executable: {sys.executable}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
            env=env
        )
        
        print("STDOUT:")
        print(result.stdout)
        print("\nSTDERR:")
        print(result.stderr)
        print(f"\nReturn code: {result.returncode}")
        
        return result.returncode
        
    except subprocess.TimeoutExpired:
        print("Test timed out after 60 seconds")
        return 1
    except Exception as e:
        print(f"Error running test: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(run_test())

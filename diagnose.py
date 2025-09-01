import sys
import os
import subprocess
from pathlib import Path

def run_command(cmd, cwd=None):
    print(f"\nRunning: {cmd}")
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True
        )
        print("Output:")
        print(result.stdout)
        if result.stderr:
            print("Error:")
            print(result.stderr)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Command failed with return code {e.returncode}")
        print("Output:")
        print(e.stdout)
        print("Error:")
        print(e.stderr)
        return False

def main():
    print("=== Python Environment Diagnostics ===\n")
    
    # Print Python version and path
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")
    print(f"Current working directory: {os.getcwd()}")
    
    # Check if we can write to the directory
    test_file = Path("test_write.tmp")
    try:
        test_file.write_text("test")
        test_file.unlink()
        print("✓ Can write to current directory")
    except Exception as e:
        print(f"✗ Cannot write to current directory: {e}")
    
    # Check basic imports
    print("\n=== Checking imports ===")
    for module in ["os", "sys", "pytest", "fastapi"]:
        try:
            __import__(module)
            print(f"✓ {module} is importable")
        except ImportError as e:
            print(f"✗ {module} is NOT importable: {e}")
    
    # Run a simple Python command
    print("\n=== Running simple Python command ===")
    run_command(f'"{sys.executable}" -c "print(\'Hello from Python!\')"')
    
    # Run pytest version
    print("\n=== Checking pytest ===")
    run_command(f'"{sys.executable}" -m pytest --version')
    
    # Run a simple test
    print("\n=== Running minimal test ===")
    test_file = Path("test_minimal.py")
    test_file.write_text('def test_pass():\n    assert True\n')
    run_command(f'"{sys.executable}" -m pytest {test_file} -v')
    test_file.unlink()
    
    print("\nDiagnostics complete!")

if __name__ == "__main__":
    main()

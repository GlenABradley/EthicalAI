import sys
import pytest

# Add the current directory to the Python path
sys.path.insert(0, '.')

# Run the test with maximum verbosity
if __name__ == "__main__":
    sys.exit(pytest.main(["tests/test_end_to_end.py::test_health_check", "-v", "-s", "--log-cli-level=DEBUG"]))

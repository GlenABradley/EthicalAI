import os
import sys
from fastapi.testclient import TestClient
from coherence.api.main import create_app

def run_test():
    print("=== Starting simple test ===")
    
    # Set up test environment
    os.environ["COHERENCE_TEST_MODE"] = "true"
    os.environ["COHERENCE_ARTIFACTS_DIR"] = "test_artifacts"
    
    try:
        # Create test client
        print("Creating FastAPI app...")
        app = create_app()
        client = TestClient(app, timeout=30.0)
        
        # Make a simple request
        print("Making request to /health/ready...")
        response = client.get("/health/ready", timeout=10.0)
        
        print(f"Response status: {response.status_code}")
        print(f"Response text: {response.text}")
        
        if response.status_code == 200:
            print("Test passed!")
            return 0
        else:
            print("Test failed!")
            return 1
            
    except Exception as e:
        print(f"Test failed with exception: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(run_test())

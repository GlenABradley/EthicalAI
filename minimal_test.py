import sys
import requests

def main():
    print("=== Minimal Test ===")
    print("Testing basic HTTP request...")
    
    try:
        response = requests.get("https://httpbin.org/get", timeout=10)
        print(f"Response status: {response.status_code}")
        print(f"Response content: {response.text[:200]}...")
        print("Test passed!")
        return 0
    except Exception as e:
        print(f"Test failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())

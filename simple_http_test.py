import sys
import requests

def test_http():
    print("Testing HTTP request to httpbin.org...")
    try:
        response = requests.get("https://httpbin.org/get", timeout=10)
        print(f"Status code: {response.status_code}")
        print(f"Response: {response.text[:200]}...")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    success = test_http()
    sys.exit(0 if success else 1)

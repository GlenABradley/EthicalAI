import numpy as np
import requests
import json

def test_resonance_with_384d_axis():
    # Create a 384-dimensional axis pack (matching the encoder output dimension)
    d = 384  # Dimension of encoder output
    k = 2     # Number of axes
    
    # Create a random orthonormal matrix Q of shape (d, k)
    # Using QR decomposition to ensure orthonormal columns
    Q = np.random.randn(d, k).astype(np.float32)
    Q, _ = np.linalg.qr(Q)
    
    # Create axis pack with the correct dimensions
    axis_pack = {
        "names": ["axis_1", "axis_2"],
        "Q": Q.tolist(),
        "lambda": [1.0, 1.0],
        "beta": [0.0, 0.0],
        "weights": [0.5, 0.5],
        "mu": {},
        "meta": {}
    }
    
    # Test with text input
    payload = {
        "texts": ["This is a test sentence."],
        "axis_pack": axis_pack,
        "return_intermediate": True,
        "encoder_name": "all-MiniLM-L6-v2"
    }
    
    print("Sending request with 384D axis pack and text input...")
    try:
        response = requests.post(
            "http://localhost:8080/resonance",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        print(f"Status code: {response.status_code}")
        print(f"Response: {response.text}")
    except Exception as e:
        print(f"Request failed: {str(e)}")
    
    # Also test with direct vectors of matching dimension
    test_vector = np.random.randn(d).astype(np.float32).tolist()
    vector_payload = {
        "vectors": [test_vector],
        "axis_pack": axis_pack,
        "return_intermediate": True
    }
    
    print("\nSending request with direct 384D vector...")
    try:
        response = requests.post(
            "http://localhost:8080/resonance",
            json=vector_payload,
            headers={"Content-Type": "application/json"}
        )
        print(f"Status code: {response.status_code}")
        print(f"Response: {response.text}")
    except Exception as e:
        print(f"Request failed: {str(e)}")

if __name__ == "__main__":
    test_resonance_with_384d_axis()

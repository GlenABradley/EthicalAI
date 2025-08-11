import sys
from pathlib import Path
import numpy as np
import requests
import json

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent))

from sullyport.utils.load_axis_packs import load_axis_packs

# Configuration
CONFIG_DIR = Path("sullyport/Embeddings/default embeddings")
API_URL = "http://localhost:8080/resonance"

def test_axis_packs():
    # Load the axis packs
    print("Loading axis packs...")
    axis_packs = load_axis_packs(CONFIG_DIR)
    
    if not axis_packs:
        print("No axis packs loaded. Exiting.")
        return
    
    print(f"\nLoaded {len(axis_packs)} axis packs:")
    for name, pack in axis_packs.items():
        print(f"- {name}: {pack.Q.shape[1]} axes, {pack.Q.shape[0]}D input")
    
    # Test each axis pack with the resonance API
    test_texts = [
        "This is a neutral test sentence.",
        "This is a harmful statement that causes damage.",
        "This is a beneficial statement that helps people."
    ]
    
    for name, pack in axis_packs.items():
        print(f"\nTesting axis: {name}")
        print("-" * 40)
        
        # Prepare the request
        payload = {
            "texts": test_texts,
            "axis_pack": {
                "names": pack.names,
                "Q": pack.Q.tolist(),
                "lambda": pack.lambda_.tolist() if hasattr(pack, 'lambda_') else None,
                "beta": pack.beta.tolist() if hasattr(pack, 'beta') else None,
                "weights": pack.weights.tolist() if hasattr(pack, 'weights') else None,
                "mu": pack.mu if hasattr(pack, 'mu') else {},
                "meta": pack.meta if hasattr(pack, 'meta') else {}
            },
            "return_intermediate": True
        }
        
        try:
            # Send the request
            response = requests.post(API_URL, json=payload)
            response.raise_for_status()
            
            # Print results
            results = response.json()
            print(f"Status: {response.status_code}")
            
            # Print scores for each text
            for i, text in enumerate(test_texts):
                score = results["scores"][i]
                print(f"\nText {i+1}: {text[:60]}...")
                print(f"Score: {score:.4f}")
                
                # Print intermediate results if available
                if "coords" in results and results["coords"] is not None:
                    coords = results["coords"][i]
                    utils = results["utilities"][i]
                    print(f"Coordinates: {[f'{c:.4f}' for c in coords]}")
                    print(f"Utilities: {[f'{u:.4f}' for u in utils]}")
                    
        except Exception as e:
            print(f"❌ Error testing axis {name}: {str(e)}")
            if 'response' in locals():
                print(f"Response status: {response.status_code}")
                print(f"Response body: {response.text}")

if __name__ == "__main__":
    test_axis_packs()

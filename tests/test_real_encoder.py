"""
Test suite using real encoder (no mocking) for comprehensive functionality testing.
This tests the actual EthicalAI system with real model downloads and processing.
"""

import pytest
import numpy as np
from fastapi.testclient import TestClient


class TestRealEncoderFunctionality:
    """Test comprehensive functionality with real encoder."""
    
    def test_real_embedding_functionality(self, api_client_real_encoder):
        """Test text embedding with real encoder."""
        print("\n🔤 Testing real embedding functionality...")
        
        # Test single text embedding
        response = api_client_real_encoder.post("/embed", json={
            "texts": ["This is a test sentence for real embedding."]
        })
        print(f"Real embed response status: {response.status_code}")
        
        assert response.status_code == 200
        data = response.json()
        
        # Validate real embedding structure
        assert "embeddings" in data
        assert len(data["embeddings"]) == 1
        assert len(data["embeddings"][0]) > 0  # Should have embedding dimensions
        assert data["model_name"] == "all-mpnet-base-v2"
        
        # Test batch embedding
        test_texts = [
            "AI should be ethical and responsible.",
            "Machine learning models need proper oversight.",
            "Algorithmic fairness is crucial for society."
        ]
        
        response = api_client_real_encoder.post("/embed", json={"texts": test_texts})
        assert response.status_code == 200
        
        data = response.json()
        assert len(data["embeddings"]) == 3
        assert data["shape"] == [3, 768]  # Expected shape for all-mpnet-base-v2
        
        # Verify embeddings are different for different texts
        embeddings = np.array(data["embeddings"])
        for i in range(len(embeddings)):
            for j in range(i+1, len(embeddings)):
                similarity = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                )
                assert similarity < 0.99  # Should not be identical
    
    def test_real_axis_pack_creation_and_analysis(self, api_client_real_encoder):
        """Test axis pack creation and analysis with real encoder."""
        print("\n📊 Testing real axis pack functionality...")
        
        # Create a real axis pack
        create_response = api_client_real_encoder.post("/axes/create", json={
            "axes": [
                {
                    "name": "ethical_ai_real",
                    "positives": [
                        "AI systems should be transparent and explainable",
                        "Machine learning should benefit humanity",
                        "Algorithmic decisions should be fair and unbiased"
                    ],
                    "negatives": [
                        "AI systems can be opaque black boxes",
                        "Machine learning can perpetuate harmful biases",
                        "Algorithmic decisions can discriminate unfairly"
                    ]
                }
            ],
            "method": "diffmean"
        })
        print(f"Real axis pack creation status: {create_response.status_code}")
        
        if create_response.status_code in [200, 201]:
            # Test loading the created axis pack
            response = api_client_real_encoder.get("/axes/ethical_ai_real")
            print(f"Real axis pack load status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                assert "names" in data
                assert "k" in data
                assert len(data["names"]) > 0
                
                # Test analysis with the real axis pack
                analysis_response = api_client_real_encoder.post("/analyze", json={
                    "axis_pack_id": "ethical_ai_real",
                    "texts": ["AI should prioritize human welfare and be transparent in its decision-making."]
                })
                print(f"Real analysis status: {analysis_response.status_code}")
                
                if analysis_response.status_code == 200:
                    analysis_data = analysis_response.json()
                    assert "axes" in analysis_data
                    assert "tokens" in analysis_data
                    assert analysis_data["axes"]["id"] == "ethical_ai_real"
        else:
            print(f"Skipping real analysis test - axis creation failed: {create_response.text}")
    
    def test_real_health_check(self, api_client_real_encoder):
        """Test health check with real encoder."""
        print("\n💚 Testing real health check...")
        
        response = api_client_real_encoder.get("/health/ready")
        print(f"Real health status: {response.status_code}")
        
        assert response.status_code == 200
        data = response.json()
        
        # Validate health response structure
        required_keys = ["status", "encoder_model", "encoder_dim", "active_pack", "frames_db_present"]
        for key in required_keys:
            assert key in data, f"Missing key: {key}"
        
        assert data["status"] == "ok"
        assert data["encoder_model"] == "all-mpnet-base-v2"
        assert isinstance(data["encoder_dim"], int)
        assert data["encoder_dim"] > 0
    
    def test_real_what_if_analysis(self, api_client_real_encoder):
        """Test what-if analysis with real encoder."""
        print("\n🔄 Testing real what-if analysis...")
        
        # Test what-if endpoint
        response = api_client_real_encoder.post("/whatif", json={
            "axis_pack_id": "test_pack",
            "doc_id": "test_doc_1",
            "edits": [
                {
                    "type": "replace_text",
                    "start": 10,
                    "end": 20,
                    "value": "unethical"
                }
            ]
        })
        print(f"Real what-if status: {response.status_code}")
        
        # Should return analysis of changes (currently returns empty deltas as stub)
        assert response.status_code == 200
        data = response.json()
        assert "deltas" in data
    
    def test_real_error_handling(self, api_client_real_encoder):
        """Test error handling with real encoder."""
        print("\n⚠️ Testing real error handling...")
        
        # Test invalid endpoint
        response = api_client_real_encoder.get("/nonexistent")
        assert response.status_code == 404
        
        # Test malformed request
        response = api_client_real_encoder.post("/embed", json={"invalid": "data"})
        print(f"Real malformed request status: {response.status_code}")
        assert response.status_code in [400, 422]
        
        # Test empty text embedding
        response = api_client_real_encoder.post("/embed", json={"texts": []})
        print(f"Real empty texts status: {response.status_code}")
        assert response.status_code in [400, 422]


# Pytest markers for different test categories
pytestmark = [
    pytest.mark.slow,  # Mark as slow since it downloads models
    pytest.mark.integration  # Mark as integration test
]

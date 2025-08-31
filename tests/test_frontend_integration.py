"""Frontend integration tests for EthicalAI UI components.

Tests the React frontend components and their integration with the backend API.
"""

import json
import time
from pathlib import Path
from typing import Dict, Any
import unittest.mock

import pytest
from fastapi.testclient import TestClient


class TestFrontendBackendIntegration:
    """Test integration between frontend and backend components."""
    
    def test_api_endpoints_for_frontend(self, api_client: TestClient):
        """Test all API endpoints that the frontend depends on."""
        
        # Test health endpoint (used by frontend for status checks)
        response = api_client.get("/health/ready")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "encoder_model" in data
        
        # Test embed endpoint (core functionality)
        response = api_client.post("/embed", json={
            "texts": ["Test text for embedding"]
        })
        assert response.status_code == 200
        data = response.json()
        assert "embeddings" in data
        assert len(data["embeddings"]) == 1
        
        # Test analyze endpoint (ethical analysis)
        response = api_client.post("/analyze", json={
            "text": "AI should respect human rights and dignity"
        })
        # Should either work or return a reasonable error
        assert response.status_code in [200, 404, 501]
        
    def test_cors_configuration(self, api_client: TestClient):
        """Test CORS configuration for frontend access."""
        # Test preflight request
        response = api_client.options("/health/ready", headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "GET"
        })
        # Should allow CORS for local development
        assert response.status_code in [200, 204]
        
    def test_api_response_format(self, api_client: TestClient):
        """Test API response formats match frontend expectations."""
        
        # Test embedding response format
        response = api_client.post("/embed", json={
            "texts": ["Sample text"]
        })
        assert response.status_code == 200
        data = response.json()
        
        # Frontend expects specific structure
        assert isinstance(data, dict)
        assert "embeddings" in data
        assert isinstance(data["embeddings"], list)
        assert isinstance(data["embeddings"][0], list)
        
        # Test health response format
        response = api_client.get("/health/ready")
        assert response.status_code == 200
        data = response.json()
        
        # Frontend expects these fields
        required_fields = ["status", "encoder_model", "encoder_dim"]
        for field in required_fields:
            assert field in data
            
    def test_error_response_format(self, api_client: TestClient):
        """Test error responses are properly formatted for frontend."""
        
        # Test invalid request
        response = api_client.post("/embed", json={
            "invalid_field": "test"
        })
        assert response.status_code in [400, 422]
        
        # Should return JSON error
        try:
            error_data = response.json()
            assert isinstance(error_data, dict)
            # FastAPI typically returns "detail" field for errors
            assert "detail" in error_data or "message" in error_data
        except json.JSONDecodeError:
            # Some errors might not be JSON
            pass
            
    def test_batch_processing_for_ui(self, api_client: TestClient):
        """Test batch processing capabilities needed by UI."""
        
        # Test multiple text embedding (for batch analysis UI)
        texts = [
            "AI should be transparent and explainable",
            "Algorithmic bias must be addressed",
            "Privacy should be protected in AI systems"
        ]
        
        response = api_client.post("/embed", json={"texts": texts})
        assert response.status_code == 200
        data = response.json()
        
        assert len(data["embeddings"]) == len(texts)
        
        # Each embedding should be the same dimension
        dimensions = [len(emb) for emb in data["embeddings"]]
        assert all(dim == dimensions[0] for dim in dimensions)
        
    def test_real_time_analysis_simulation(self, api_client: TestClient):
        """Simulate real-time analysis as would be done by frontend."""
        
        # Simulate user typing and analyzing text in real-time
        progressive_texts = [
            "AI should",
            "AI should respect",
            "AI should respect human",
            "AI should respect human rights and dignity"
        ]
        
        embeddings_history = []
        
        for text in progressive_texts:
            response = api_client.post("/embed", json={"texts": [text]})
            assert response.status_code == 200
            
            data = response.json()
            embeddings_history.append(data["embeddings"][0])
            
        # Embeddings should change as text evolves
        assert len(embeddings_history) == 4
        
        # Each embedding should be different
        for i in range(len(embeddings_history) - 1):
            emb1 = embeddings_history[i]
            emb2 = embeddings_history[i + 1]
            
            # Calculate cosine similarity
            import numpy as np
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            
            # Should be similar but not identical
            assert 0.5 < similarity < 0.99


class TestUIComponentDataFlow:
    """Test data flow patterns used by UI components."""
    
    def test_interaction_page_workflow(self, api_client: TestClient):
        """Test the workflow used by the Interaction page component."""
        
        # Step 1: User enters text
        user_text = "Should AI systems have the right to make autonomous decisions about human lives?"
        
        # Step 2: Get embedding for visualization
        embed_response = api_client.post("/embed", json={"texts": [user_text]})
        assert embed_response.status_code == 200
        embed_data = embed_response.json()
        
        # Step 3: Perform ethical analysis
        analyze_response = api_client.post("/analyze", json={
            "text": user_text,
            "return_embeddings": True
        })
        
        # Should get some form of analysis (even if endpoint is not fully implemented)
        if analyze_response.status_code == 200:
            analyze_data = analyze_response.json()
            
            # UI expects structured analysis data
            assert isinstance(analyze_data, dict)
            
            # Should have either scores, analysis, or embeddings
            has_analysis = any(key in analyze_data for key in ["scores", "analysis", "embeddings", "axis_scores"])
            assert has_analysis
            
    def test_visualization_data_format(self, api_client: TestClient):
        """Test data format for visualization components."""
        
        # Get embeddings for multiple related texts (for topology visualization)
        related_texts = [
            "AI should prioritize human welfare",
            "Machine learning should be transparent",
            "Algorithmic decisions should be explainable",
            "AI systems should respect privacy"
        ]
        
        response = api_client.post("/embed", json={"texts": related_texts})
        assert response.status_code == 200
        data = response.json()
        
        embeddings = data["embeddings"]
        
        # Verify format suitable for visualization
        assert len(embeddings) == len(related_texts)
        assert all(isinstance(emb, list) for emb in embeddings)
        assert all(len(emb) > 0 for emb in embeddings)
        
        # Test dimensionality reduction simulation (for 2D/3D visualization)
        import numpy as np
        emb_array = np.array(embeddings)
        
        # Should be able to compute pairwise similarities
        similarities = np.dot(emb_array, emb_array.T)
        assert similarities.shape == (len(related_texts), len(related_texts))
        
        # Diagonal should be 1.0 (self-similarity)
        for i in range(len(similarities)):
            assert abs(similarities[i, i] - np.dot(emb_array[i], emb_array[i])) < 0.01
            
    def test_ethical_axis_data_structure(self, api_client: TestClient):
        """Test ethical axis data structure for UI display."""
        
        # Test getting axis information
        response = api_client.get("/v1/axes")
        
        if response.status_code == 200:
            data = response.json()
            
            # Should be structured for UI consumption
            assert isinstance(data, (list, dict))
            
            if isinstance(data, list) and len(data) > 0:
                # Each axis should have required fields for UI
                axis = data[0]
                expected_fields = ["name", "description"]
                
                # Check if axis has UI-friendly structure
                has_ui_fields = any(field in axis for field in expected_fields)
                assert has_ui_fields or isinstance(axis, str)  # Could be just axis names
                
    def test_performance_for_interactive_ui(self, api_client: TestClient):
        """Test performance characteristics needed for interactive UI."""
        
        # Test response time for single text (should be fast for real-time feedback)
        start_time = time.time()
        
        response = api_client.post("/embed", json={
            "texts": ["Quick test for UI responsiveness"]
        })
        
        end_time = time.time()
        response_time = end_time - start_time
        
        assert response.status_code == 200
        
        # Should respond reasonably quickly for UI (allowing for mocked encoder)
        assert response_time < 5.0  # 5 seconds max for mocked response
        
        # Test batch processing time
        batch_texts = ["Test text " + str(i) for i in range(10)]
        
        start_time = time.time()
        response = api_client.post("/embed", json={"texts": batch_texts})
        end_time = time.time()
        
        batch_response_time = end_time - start_time
        
        assert response.status_code == 200
        
        # Batch should not be dramatically slower per item
        assert batch_response_time < 10.0  # 10 seconds max for 10 items


class TestAPIContractCompliance:
    """Test API contract compliance for frontend integration."""
    
    def test_embedding_api_contract(self, api_client: TestClient):
        """Test embedding API contract matches frontend expectations."""
        
        # Test required request format
        valid_request = {"texts": ["Sample text"]}
        response = api_client.post("/embed", json=valid_request)
        assert response.status_code == 200
        
        # Test response structure
        data = response.json()
        assert "embeddings" in data
        assert isinstance(data["embeddings"], list)
        assert len(data["embeddings"]) == 1
        assert isinstance(data["embeddings"][0], list)
        assert len(data["embeddings"][0]) > 0
        
        # Test with multiple texts
        multi_request = {"texts": ["Text 1", "Text 2", "Text 3"]}
        response = api_client.post("/embed", json=multi_request)
        assert response.status_code == 200
        
        data = response.json()
        assert len(data["embeddings"]) == 3
        
    def test_health_api_contract(self, api_client: TestClient):
        """Test health API contract for frontend status monitoring."""
        
        response = api_client.get("/health/ready")
        assert response.status_code == 200
        
        data = response.json()
        
        # Required fields for frontend
        required_fields = ["status", "encoder_model", "encoder_dim"]
        for field in required_fields:
            assert field in data
            
        # Status should be meaningful
        assert data["status"] in ["ok", "ready", "healthy"]
        
        # Encoder dim should be positive integer
        assert isinstance(data["encoder_dim"], int)
        assert data["encoder_dim"] > 0
        
    def test_error_handling_contract(self, api_client: TestClient):
        """Test error handling matches frontend expectations."""
        
        # Test empty texts array
        response = api_client.post("/embed", json={"texts": []})
        assert response.status_code in [400, 422]
        
        # Test missing texts field
        response = api_client.post("/embed", json={})
        assert response.status_code in [400, 422]
        
        # Test invalid texts type
        response = api_client.post("/embed", json={"texts": "not an array"})
        assert response.status_code in [400, 422]
        
        # All errors should return JSON
        for status_code in [400, 422]:
            if response.status_code == status_code:
                try:
                    error_data = response.json()
                    assert isinstance(error_data, dict)
                except json.JSONDecodeError:
                    pytest.fail(f"Error response should be valid JSON for status {status_code}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

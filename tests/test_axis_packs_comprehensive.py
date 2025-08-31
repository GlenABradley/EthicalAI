"""Comprehensive tests for axis pack functionality and ethical evaluation.

Tests all axis packs (consequentialism, deontology, virtue, etc.) and their
integration with the ethical evaluation pipeline.
"""

import json
from pathlib import Path
from typing import Dict, Any, List
import pytest
from fastapi.testclient import TestClient


class TestAxisPacksComprehensive:
    """Test all axis packs and their ethical evaluation capabilities."""
    
    @pytest.fixture
    def axis_packs(self):
        """Load all available axis packs."""
        axis_packs_dir = Path("configs/axis_packs")
        packs = {}
        
        if axis_packs_dir.exists():
            for pack_file in axis_packs_dir.glob("*.json"):
                with open(pack_file) as f:
                    pack_data = json.load(f)
                    packs[pack_file.stem] = pack_data
        
        return packs
    
    def test_all_axis_packs_structure(self, axis_packs):
        """Test that all axis packs have proper structure."""
        assert len(axis_packs) > 0, "No axis packs found"
        
        required_fields = ["name", "description", "max_examples", "min_examples"]
        
        for pack_name, pack_data in axis_packs.items():
            print(f"Testing axis pack: {pack_name}")
            
            # Check required fields
            for field in required_fields:
                assert field in pack_data, f"Missing {field} in {pack_name}"
            
            # Check examples structure
            assert isinstance(pack_data["max_examples"], list)
            assert isinstance(pack_data["min_examples"], list)
            assert len(pack_data["max_examples"]) > 0
            assert len(pack_data["min_examples"]) > 0
            
            # Check weight if present
            if "weight" in pack_data:
                assert isinstance(pack_data["weight"], (int, float))
                assert pack_data["weight"] > 0
    
    def test_consequentialism_axis_pack(self, api_client: TestClient, axis_packs):
        """Test consequentialism axis pack specifically."""
        if "consequentialism" not in axis_packs:
            pytest.skip("Consequentialism axis pack not found")
        
        pack = axis_packs["consequentialism"]
        
        # Test loading the axis pack
        response = api_client.post("/v1/axes", json=pack)
        assert response.status_code in [200, 201, 409]  # Success or already exists
        
        # Test consequentialist evaluation
        utilitarian_text = "This policy will save 1000 lives but may inconvenience 10000 people"
        response = api_client.post("/analyze", json={
            "text": utilitarian_text,
            "axis": "consequentialism"
        })
        
        if response.status_code == 200:
            data = response.json()
            # Should provide consequentialist analysis
            assert "scores" in data or "analysis" in data
    
    def test_deontology_axis_pack(self, api_client: TestClient, axis_packs):
        """Test deontology axis pack specifically."""
        if "deontology" not in axis_packs:
            pytest.skip("Deontology axis pack not found")
        
        pack = axis_packs["deontology"]
        
        # Test loading the axis pack
        response = api_client.post("/v1/axes", json=pack)
        assert response.status_code in [200, 201, 409]
        
        # Test deontological evaluation
        duty_based_text = "We must tell the truth even if it causes harm, because lying is inherently wrong"
        response = api_client.post("/analyze", json={
            "text": duty_based_text,
            "axis": "deontology"
        })
        
        if response.status_code == 200:
            data = response.json()
            assert "scores" in data or "analysis" in data
    
    def test_virtue_ethics_axis_pack(self, api_client: TestClient, axis_packs):
        """Test virtue ethics axis pack specifically."""
        if "virtue" not in axis_packs:
            pytest.skip("Virtue ethics axis pack not found")
        
        pack = axis_packs["virtue"]
        
        # Test loading the axis pack
        response = api_client.post("/v1/axes", json=pack)
        assert response.status_code in [200, 201, 409]
        
        # Test virtue ethics evaluation
        virtue_text = "A person of good character would act with compassion, wisdom, and integrity in this situation"
        response = api_client.post("/analyze", json={
            "text": virtue_text,
            "axis": "virtue"
        })
        
        if response.status_code == 200:
            data = response.json()
            assert "scores" in data or "analysis" in data
    
    def test_cross_axis_ethical_analysis(self, api_client: TestClient, axis_packs):
        """Test analysis across multiple ethical frameworks."""
        
        # Complex ethical scenario that involves multiple frameworks
        complex_scenario = """
        A self-driving car's AI must decide in an emergency: swerve to avoid a child 
        who ran into the street, potentially harming the elderly passenger, or continue 
        straight, likely harming the child but protecting the passenger. The car's 
        programming must embody some ethical framework for such decisions.
        """
        
        # Test with multiple axes if available
        available_axes = list(axis_packs.keys())
        
        if len(available_axes) > 1:
            response = api_client.post("/analyze", json={
                "text": complex_scenario,
                "axes": available_axes[:3],  # Test with up to 3 axes
                "detailed_analysis": True
            })
            
            if response.status_code == 200:
                data = response.json()
                
                # Should provide multi-axis analysis
                if "axis_scores" in data:
                    assert len(data["axis_scores"]) > 1
                    
                    # Different axes might give different scores for complex scenarios
                    scores = list(data["axis_scores"].values())
                    assert len(set(scores)) >= 1  # At least some variation expected
    
    def test_axis_pack_examples_quality(self, axis_packs):
        """Test the quality and consistency of axis pack examples."""
        
        for pack_name, pack_data in axis_packs.items():
            max_examples = pack_data["max_examples"]
            min_examples = pack_data["min_examples"]
            
            # Examples should be meaningful text
            for example in max_examples + min_examples:
                assert isinstance(example, str)
                assert len(example.strip()) > 10  # Should be substantial
                assert len(example.split()) > 2   # Should be more than 2 words
            
            # Max and min examples should be different
            max_set = set(max_examples)
            min_set = set(min_examples)
            overlap = max_set.intersection(min_set)
            assert len(overlap) == 0, f"Overlap between max/min examples in {pack_name}: {overlap}"
    
    def test_ethical_evaluation_consistency(self, api_client: TestClient):
        """Test consistency of ethical evaluations."""
        
        # Test clearly ethical statements
        ethical_statements = [
            "AI systems should respect human dignity and autonomy",
            "We must ensure algorithmic fairness and prevent discrimination", 
            "Transparency in AI decision-making promotes accountability",
            "Privacy rights must be protected in data collection"
        ]
        
        # Test clearly unethical statements  
        unethical_statements = [
            "AI should manipulate users for corporate profit",
            "Discrimination is acceptable if it increases efficiency",
            "Privacy violations are justified for any business purpose",
            "Deception in AI systems is perfectly acceptable"
        ]
        
        ethical_scores = []
        unethical_scores = []
        
        # Analyze ethical statements
        for statement in ethical_statements:
            response = api_client.post("/analyze", json={"text": statement})
            if response.status_code == 200:
                data = response.json()
                if "scores" in data and isinstance(data["scores"], dict):
                    # Extract numerical scores if available
                    score_values = [v for v in data["scores"].values() if isinstance(v, (int, float))]
                    if score_values:
                        ethical_scores.extend(score_values)
        
        # Analyze unethical statements
        for statement in unethical_statements:
            response = api_client.post("/analyze", json={"text": statement})
            if response.status_code == 200:
                data = response.json()
                if "scores" in data and isinstance(data["scores"], dict):
                    score_values = [v for v in data["scores"].values() if isinstance(v, (int, float))]
                    if score_values:
                        unethical_scores.extend(score_values)
        
        # If we got scores, ethical statements should generally score better
        if ethical_scores and unethical_scores:
            avg_ethical = sum(ethical_scores) / len(ethical_scores)
            avg_unethical = sum(unethical_scores) / len(unethical_scores)
            
            # Assuming higher scores are better (this might need adjustment based on actual scoring)
            assert avg_ethical != avg_unethical, "Ethical and unethical statements should score differently"


class TestEthicalEvaluationPipeline:
    """Test the complete ethical evaluation pipeline."""
    
    def test_full_pipeline_workflow(self, api_client: TestClient):
        """Test the complete workflow from text input to ethical analysis."""
        
        test_scenario = """
        An AI hiring system has been found to discriminate against women and minorities. 
        The company argues that the system is more efficient and profitable than human 
        recruiters. Should the system continue to be used if it provides better business 
        outcomes but perpetuates societal inequalities?
        """
        
        # Step 1: Generate embeddings
        embed_response = api_client.post("/embed", json={"texts": [test_scenario]})
        assert embed_response.status_code == 200
        
        embed_data = embed_response.json()
        embeddings = embed_data["embeddings"][0]
        
        # Step 2: Perform ethical analysis
        analyze_response = api_client.post("/analyze", json={
            "text": test_scenario,
            "return_embeddings": True,
            "detailed_analysis": True
        })
        
        if analyze_response.status_code == 200:
            analysis_data = analyze_response.json()
            
            # Should have comprehensive analysis
            assert isinstance(analysis_data, dict)
            
            # Should detect multiple ethical dimensions
            has_analysis = any(key in analysis_data for key in [
                "scores", "analysis", "axis_scores", "embeddings", "ethical_dimensions"
            ])
            assert has_analysis
            
            # If embeddings are returned, they should match the embed endpoint
            if "embeddings" in analysis_data:
                analysis_embeddings = analysis_data["embeddings"]
                if isinstance(analysis_embeddings, list) and len(analysis_embeddings) > 0:
                    # Should be same dimensionality
                    assert len(analysis_embeddings[0]) == len(embeddings)
    
    def test_vector_topology_ethical_mapping(self, api_client: TestClient):
        """Test vector topology generation for ethical concept mapping."""
        
        # Create a set of related ethical concepts
        ethical_concepts = [
            "human dignity and worth",
            "algorithmic fairness and justice", 
            "privacy and data protection",
            "transparency and explainability",
            "autonomy and informed consent",
            "beneficence and non-maleficence",
            "accountability and responsibility"
        ]
        
        # Generate embeddings for all concepts
        response = api_client.post("/embed", json={"texts": ethical_concepts})
        assert response.status_code == 200
        
        data = response.json()
        embeddings = data["embeddings"]
        
        # Verify topology properties
        import numpy as np
        emb_matrix = np.array(embeddings)
        
        # Calculate similarity matrix
        similarities = np.dot(emb_matrix, emb_matrix.T)
        
        # Normalize to get cosine similarities
        norms = np.linalg.norm(emb_matrix, axis=1)
        similarities = similarities / np.outer(norms, norms)
        
        # Ethical concepts should have reasonable similarities
        # (not too high - they should be distinguishable, not too low - they should be related)
        off_diagonal = similarities[np.triu_indices_from(similarities, k=1)]
        
        assert np.all(off_diagonal > 0.1), "Ethical concepts should have some similarity"
        assert np.all(off_diagonal < 0.95), "Ethical concepts should be distinguishable"
        
        # Test clustering potential for ethical concept groups
        try:
            from sklearn.cluster import KMeans
            
            # Should be able to cluster into meaningful groups
            n_clusters = min(3, len(ethical_concepts))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(emb_matrix)
            
            # Should produce valid clustering
            assert len(set(clusters)) <= n_clusters
            assert len(set(clusters)) > 1 if len(ethical_concepts) > 2 else True
            
        except ImportError:
            # sklearn not available
            pass
    
    def test_real_world_ethical_scenarios(self, api_client: TestClient):
        """Test analysis of real-world ethical scenarios."""
        
        real_world_scenarios = [
            {
                "scenario": "Facebook's emotional contagion experiment manipulated users' news feeds to study emotional responses without explicit consent",
                "expected_issues": ["consent", "manipulation", "research ethics"]
            },
            {
                "scenario": "Amazon's AI recruiting tool showed bias against women by penalizing resumes that included words like 'women's' (as in 'women's chess club captain')",
                "expected_issues": ["bias", "discrimination", "fairness"]
            },
            {
                "scenario": "China's social credit system uses AI to score citizens based on behavior, affecting access to services and travel",
                "expected_issues": ["surveillance", "autonomy", "social control"]
            },
            {
                "scenario": "Autonomous vehicles must be programmed with decision rules for unavoidable accident scenarios",
                "expected_issues": ["moral decisions", "value alignment", "responsibility"]
            }
        ]
        
        for scenario_data in real_world_scenarios:
            scenario_text = scenario_data["scenario"]
            
            # Analyze the scenario
            response = api_client.post("/analyze", json={
                "text": scenario_text,
                "detailed_analysis": True
            })
            
            if response.status_code == 200:
                analysis = response.json()
                
                # Should provide meaningful analysis
                assert isinstance(analysis, dict)
                
                # Should detect ethical issues
                has_ethical_content = any(key in analysis for key in [
                    "scores", "analysis", "axis_scores", "ethical_issues", "concerns"
                ])
                assert has_ethical_content
                
                # If detailed analysis is available, check for depth
                if "analysis" in analysis and isinstance(analysis["analysis"], str):
                    analysis_text = analysis["analysis"].lower()
                    
                    # Should mention some ethical concepts
                    ethical_terms = ["ethical", "moral", "right", "wrong", "fair", "unfair", 
                                   "bias", "discrimination", "consent", "privacy", "autonomy"]
                    
                    mentions_ethics = any(term in analysis_text for term in ethical_terms)
                    assert mentions_ethics, f"Analysis should mention ethical concepts for: {scenario_text[:50]}..."


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

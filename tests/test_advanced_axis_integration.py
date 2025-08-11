"""Integration test for advanced axis builder with resonance evaluator."""

import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import logging

from coherence.axis.advanced_builder import AdvancedAxisBuilder, build_advanced_axis_pack
from coherence.axis.pack import AxisPack
from coherence.metrics.resonance import resonance
from coherence.encoders.text_sbert import get_default_encoder

# Set up logging with debug level
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Also set debug level for coherence modules
logging.getLogger('coherence').setLevel(logging.DEBUG)

def test_advanced_axis_resonance():
    """Test resonance scoring with advanced axis pack."""
    # Sample axis configuration
    axis_configs = [
        {
            "name": "positive_negative",
            "max_examples": ["This is great", "I love this", "Amazing experience"],
            "min_examples": ["This is terrible", "I hate this", "Worst experience"],
            "description": "Positive vs negative sentiment"
        },
        {
            "name": "formal_informal",
            "max_examples": ["The meeting has been rescheduled", "Please find attached", "I would appreciate your feedback"],
            "min_examples": ["Meeting's moved", "Here's the file", "Let me know what you think"],
            "description": "Formal vs informal language"
        }
    ]
    
    # Create temporary JSON files for testing
    json_paths = []
    try:
        for i, config in enumerate(axis_configs):
            path = f"test_axis_{i}.json"
            with open(path, 'w') as f:
                json.dump(config, f)
            json_paths.append(path)
        
        # Initialize encoder
        encoder = get_default_encoder()
        
        # Build axis pack using advanced builder
        axis_pack = build_advanced_axis_pack(
            json_paths=json_paths,
            encode_fn=encoder.encode,
            whitening=True,
            whitening_method='empirical',
            use_lda=True,
            margin_alpha=0.7,
            orthogonalize=True,
            n_bootstrap=10,
            random_state=42
        )
        
        # Test with some sample texts
        test_texts = [
            "This product is absolutely fantastic!",  # Should be positive
            "I really dislike this approach",        # Should be negative
            "The meeting has been rescheduled to tomorrow"  # Should be formal
        ]
        
        # Encode test texts
        test_vectors = encoder.encode(test_texts)
        
        # Get resonance scores
        logger.info("Calling resonance function...")
        scores = resonance(test_vectors, axis_pack)
        
        # Log detailed information about the scores
        logger.info(f"Scores type: {type(scores)}")
        logger.info(f"Scores shape: {scores.shape if hasattr(scores, 'shape') else 'N/A'}")
        logger.info(f"Scores dtype: {scores.dtype if hasattr(scores, 'dtype') else 'N/A'}")
        logger.info(f"Scores content: {scores}")
        
        logger.info("Test texts and scores:")
        for i, (text, score) in enumerate(zip(test_texts, scores)):
            logger.info(f"Text {i}: {text}")
            logger.info(f"  Score type: {type(score)}")
            logger.info(f"  Score value: {score}")
        
        # Basic validation
        logger.info("Running validations...")
        assert isinstance(axis_pack, AxisPack), "Should return an AxisPack instance"
        assert len(scores) == len(test_texts), f"Should return one score per input text. Got {len(scores)} scores for {len(test_texts)} texts"
        
        # Ensure scores are finite
        if hasattr(scores, '__iter__'):
            for i, score in enumerate(scores):
                assert np.isfinite(score), f"Score at index {i} is not finite: {score}"
        else:
            assert np.isfinite(scores), f"Score is not finite: {scores}"
        
        # The first text should be more positive than the second
        score1 = float(scores[0]) if hasattr(scores, '__getitem__') else float(scores)
        score2 = float(scores[1]) if hasattr(scores, '__getitem__') else float(scores)
        logger.info(f"Comparing scores: {score1} > {score2}")
        assert score1 > score2, \
            f"Positive text should have higher score than negative text. Got: {score1} vs {score2}"
        
        logger.info("All tests passed!")
        
    finally:
        # Clean up temporary files
        for path in json_paths:
            try:
                Path(path).unlink()
            except Exception as e:
                logger.warning(f"Failed to delete {path}: {e}")

if __name__ == "__main__":
    test_advanced_axis_resonance()

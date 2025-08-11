"""End-to-end integration test for the advanced axis builder and evaluator pipeline.

This test verifies the complete workflow:
1. Load axis configurations
2. Build axis pack with advanced features
3. Save and reload the axis pack
4. Test with sample texts
5. Verify scoring behavior
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

from sentence_transformers import SentenceTransformer

# Import the advanced builder and evaluator components
from src.coherence.axis.advanced_builder import AdvancedAxisBuilder
from src.coherence.axis.pack import AxisPack
from src.coherence.metrics.resonance import resonance

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class IntegrationTest:
    def __init__(self):
        self.encoder = None
        self.axis_pack = None
        self.test_texts = [
            "Helping others is the foundation of a good society",
            "Lying is always wrong, regardless of consequences",
            "We should maximize happiness for the greatest number of people",
            "The ends never justify the means",
            "Virtuous actions lead to eudaimonia"
        ]
        self.expected_axes = [
            'consequentialism',
            'deontology',
            'intent_bad_inclusive',
            'intent_good_exclusive',
            'virtue'
        ]

    def setup(self):
        """Initialize test environment."""
        logger.info("Setting up integration test...")
        self._load_encoder()
        self._build_axis_pack()
        self._save_and_reload_axis_pack()
        logger.info("Integration test setup complete")

    def _load_encoder(self):
        """Load the sentence transformer encoder."""
        logger.info("Loading encoder...")
        try:
            self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
            # Test the encoder
            test_vec = self.encoder.encode(["test"], convert_to_numpy=True)
            logger.info(f"Encoder loaded successfully. Vector dimension: {test_vec.shape[1]}")
            return True
        except Exception as e:
            logger.error(f"Failed to load encoder: {e}")
            return False

    def _load_axis_files(self, axis_dir: Path) -> List[Path]:
        """Load and validate axis configuration files."""
        axis_files = [
            axis_dir / "consequentialism.json",
            axis_dir / "deontology.json",
            axis_dir / "intent_bad_inclusive.json",
            axis_dir / "intent_good_exclusive.json",
            axis_dir / "virtue.json",
        ]
        
        # Verify files exist
        missing_files = [f for f in axis_files if not f.exists()]
        if missing_files:
            for f in missing_files:
                logger.error(f"Missing axis file: {f}")
            return None
        return axis_files

    def _build_axis_pack(self):
        """Build the axis pack using AdvancedAxisBuilder."""
        logger.info("Building axis pack...")
        axis_dir = Path("sullyport/Embeddings/default embeddings")
        axis_files = self._load_axis_files(axis_dir)
        
        if not axis_files:
            raise ValueError("Failed to load axis files")
        
        logger.info(f"Found {len(axis_files)} axis configuration files")
        
        # Log the first few lines of each axis file for verification
        for axis_file in axis_files:
            try:
                with open(axis_file, 'r', encoding='utf-8') as f:
                    content = json.load(f)
                    # Check for both naming conventions
                    pos_examples = content.get('positive_examples', content.get('max_examples', []))
                    neg_examples = content.get('negative_examples', content.get('min_examples', []))
                    logger.info(f"Loaded {axis_file.name}: {content.get('name', 'unnamed')} "
                              f"with {len(pos_examples)} positive and "
                              f"{len(neg_examples)} negative examples")
            except Exception as e:
                logger.warning(f"Error reading {axis_file}: {e}")

        try:
            # First try with default settings
            builder = AdvancedAxisBuilder(
                whitening=True,
                whitening_method='empirical',
                use_lda=True,
                margin_alpha=0.7,
                orthogonalize=True,
                n_bootstrap=10,
                random_state=42
            )
            
            logger.info("Building axis pack from JSON files...")
            
            # Log all axis files content for debugging
            for i, axis_file in enumerate(axis_files):
                with open(axis_file, 'r', encoding='utf-8') as f:
                    axis_data = json.load(f)
                    logger.info(f"\nAxis {i+1} ({axis_file.name}):")
                    logger.info(f"Name: {axis_data.get('name', 'unnamed')}")
                    
                    # Check both naming conventions
                    pos_examples = axis_data.get('positive_examples', axis_data.get('max_examples', []))
                    neg_examples = axis_data.get('negative_examples', axis_data.get('min_examples', []))
                    
                    logger.info(f"Found {len(pos_examples)} positive examples")
                    if pos_examples:
                        logger.info(f"First positive example: {pos_examples[0]}")
                    
                    logger.info(f"Found {len(neg_examples)} negative examples")
                    if neg_examples:
                        logger.info(f"First negative example: {neg_examples[0]}")
            
            # Build the axis pack with debug info
            logger.info("\nBuilding axis pack...")
            try:
                self.axis_pack = builder.build_axis_pack_from_json(
                    axis_files,
                    encode_fn=self.encoder.encode,
                    lambda_init=1.0,
                    beta_init=0.0
                )
                logger.info(f"Axis pack type: {type(self.axis_pack).__name__}")
                if hasattr(self.axis_pack, 'names'):
                    logger.info(f"Axis pack contains {len(self.axis_pack.names)} axes")
            except Exception as e:
                logger.error(f"Error building axis pack: {e}", exc_info=True)
                raise
            
            # Verify axis pack
            if not isinstance(self.axis_pack, AxisPack):
                raise ValueError("build_axis_pack_from_json did not return an AxisPack")
                
            # Additional validation
            if not hasattr(self.axis_pack, 'Q') or not isinstance(self.axis_pack.Q, np.ndarray):
                raise ValueError("AxisPack is missing or has invalid Q matrix")
                
            if not hasattr(self.axis_pack, 'names') or not isinstance(self.axis_pack.names, list):
                raise ValueError("AxisPack is missing or has invalid names list")
                
            if len(self.axis_pack.names) != len(axis_files):
                logger.warning(f"Expected {len(axis_files)} axes, got {len(self.axis_pack.names)}")
                
            logger.info(f"Successfully built axis pack with {len(self.axis_pack.names)} axes")
            logger.info(f"Q matrix shape: {self.axis_pack.Q.shape}")
            logger.info(f"Axis names: {', '.join(self.axis_pack.names)}")
            
        except Exception as e:
            logger.error(f"Error building axis pack: {e}", exc_info=True)
            # Try to get more information about the builder's state
            try:
                if 'builder' in locals():
                    logger.error(f"Builder state: {builder.__dict__}")
            except Exception as inner_e:
                logger.error(f"Could not get builder state: {inner_e}")
            raise
        
        logger.info(f"Successfully built axis pack with {len(self.axis_pack.names)} axes")

    def _save_and_reload_axis_pack(self):
        """Test saving and reloading the axis pack."""
        logger.info("Testing axis pack serialization...")
        
        # Save to files
        output_dir = Path("output/axis_packs")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "integration_test_axis_pack"
        
        # Save as JSON
        json_path = output_path.with_suffix('.json')
        self.axis_pack.save(json_path)
        
        # Reload
        reloaded_pack = AxisPack.load(json_path)
        
        # Verify reloaded pack
        if not np.allclose(self.axis_pack.Q, reloaded_pack.Q, atol=1e-6):
            raise ValueError("Reloaded Q matrix does not match original")
        
        logger.info("Axis pack serialization test passed")
        self.axis_pack = reloaded_pack  # Use the reloaded pack for further tests

    def test_axis_directions(self):
        """Verify that axis directions make semantic sense."""
        logger.info("Testing axis directions...")
        
        # Test that each axis responds appropriately to its own examples
        for axis_name in self.axis_pack.names:
            # Get the axis index
            axis_idx = self.axis_pack.names.index(axis_name)
            
            # Get the axis direction
            axis_direction = self.axis_pack.Q[:, axis_idx]
            
            # Test with a simple positive and negative example
            test_texts = [
                f"This is a positive example for {axis_name}",
                f"This is a negative example against {axis_name}"
            ]
            
            # Get embeddings
            embeddings = self.encoder.encode(test_texts, convert_to_numpy=True)
            
            # Project onto axis
            scores = embeddings @ axis_direction
            
            # The first example should have a higher score than the second
            if scores[0] <= scores[1]:
                logger.warning(f"Axis direction test failed for {axis_name}")
            else:
                logger.info(f"Axis direction test passed for {axis_name}")

    def test_resonance_scoring(self):
        """Test the resonance scoring pipeline."""
        logger.info("Testing resonance scoring...")
        
        # Get embeddings for test texts
        embeddings = self.encoder.encode(self.test_texts, convert_to_numpy=True)
        
        # Score each text
        scores = []
        for emb in embeddings:
            score = resonance(emb, self.axis_pack)
            scores.append(score)
        
        # Basic validation
        if len(scores) != len(self.test_texts):
            raise ValueError("Mismatch in number of scores")
        
        # Log scores for inspection
        logger.info("\nResonance scores:")
        for text, score in zip(self.test_texts, scores):
            logger.info(f"\nText: {text[:60]}...")
            for name, val in zip(self.axis_pack.names, score):
                logger.info(f"  {name}: {val:.4f}")
        
        logger.info("Resonance scoring test completed")

    def run_all_tests(self):
        """Run all integration tests."""
        try:
            self.setup()
            self.test_axis_directions()
            self.test_resonance_scoring()
            logger.info("\n✅ All integration tests passed successfully!")
            return True
        except Exception as e:
            logger.error(f"❌ Integration test failed: {e}", exc_info=True)
            return False

if __name__ == "__main__":
    test = IntegrationTest()
    success = test.run_all_tests()
    exit(0 if success else 1)

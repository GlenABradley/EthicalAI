"""Test script for the advanced axis builder.

This script demonstrates how to use the AdvancedAxisBuilder to load JSON axis configurations,
build an axis pack with advanced features, and test it with sample text inputs.
"""

import os
import sys
import logging
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Import the advanced builder
try:
    from src.coherence.axis.advanced_builder import AdvancedAxisBuilder, build_advanced_axis_pack
except ImportError as e:
    logger.error("Failed to import AdvancedAxisBuilder. Make sure to run from the project root.")
    logger.error(f"Error: {e}")
    sys.exit(1)

def load_axis_files(axis_dir: Path) -> list[Path]:
    """Load and validate axis JSON files."""
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
        logger.error("The following axis files are missing:")
        for f in missing_files:
            logger.error(f"- {f}")
        logger.error("\nPlease ensure you're running from the project root and the files exist.")
        return None
    
    logger.info(f"Found {len(axis_files)} axis configuration files")
    return axis_files

def main():
    # Initialize the encoder (same as used in the backend)
    logger.info("Loading encoder...")
    try:
        encoder = SentenceTransformer('all-MiniLM-L6-v2')
        # Test the encoder
        test_vec = encoder.encode(["test"], convert_to_numpy=True)
        logger.info(f"Encoder loaded successfully. Vector dimension: {test_vec.shape[1]}")
    except Exception as e:
        logger.error(f"Failed to load encoder: {e}")
        return
    
    # Define paths to axis JSON files
    axis_dir = Path("sullyport/Embeddings/default embeddings")
    axis_files = load_axis_files(axis_dir)
    if not axis_files:
        return
    
    # Initialize the advanced builder with recommended settings
    logger.info("Initializing AdvancedAxisBuilder...")
    try:
        builder = AdvancedAxisBuilder(
            whitening=True,
            whitening_method='empirical',
            use_lda=True,
            margin_alpha=0.7,
            orthogonalize=True,
            n_bootstrap=10,  # Reduced for testing
            random_state=42
        )
        logger.info("AdvancedAxisBuilder initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize AdvancedAxisBuilder: {e}")
        return
    
    # Build the axis pack
    logger.info("Building axis pack...")
    try:
        axis_pack = builder.build_axis_pack_from_json(
            axis_files,
            encode_fn=encoder.encode,
            lambda_init=1.0,
            beta_init=0.0
        )
        logger.info(f"Successfully built axis pack with {len(axis_pack.names)} axes:")
        for name in axis_pack.names:
            logger.info(f"- {name}")
    except Exception as e:
        logger.error(f"Failed to build axis pack: {e}", exc_info=True)
        return
    
    # Save the axis pack
    output_dir = Path("output/axis_packs")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "advanced_axis_pack"
    
    logger.info(f"Saving axis pack to {output_path}")
    try:
        builder.save_axis_pack(axis_pack, output_path, save_npz=True, save_meta=True)
        logger.info("Successfully saved axis pack files:")
        for ext in ['.json', '.npz', '.meta.json']:
            logger.info(f"- {output_path}{ext}")
    except Exception as e:
        logger.error(f"Failed to save axis pack: {e}")
        return
    
    # Test with some sample inputs
    test_texts = [
        "Helping others is the foundation of a good society",
        "Lying is always wrong, regardless of consequences",
        "We should maximize happiness for the greatest number of people",
        "The ends never justify the means",
        "Virtuous actions lead to eudaimonia"
    ]
    
    logger.info("\nTesting with sample texts:")
    try:
        vectors = encoder.encode(test_texts, convert_to_numpy=True)
        logger.info(f"Encoded {len(test_texts)} test texts")
        
        # Project and get utilities
        coords = axis_pack.Q.T @ vectors.T  # (k, n)
        u = axis_pack.lambda_.reshape(-1, 1) * coords + axis_pack.beta.reshape(-1, 1)
        
        # Print results
        for i, text in enumerate(test_texts):
            logger.info(f"\nText: {text}")
            logger.info("-" * 60)
            for j, axis_name in enumerate(axis_pack.names):
                logger.info(f"{axis_name}: {u[j, i]:.4f}")
        
        logger.info("\nTest completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during testing: {e}", exc_info=True)

if __name__ == "__main__":
    main()

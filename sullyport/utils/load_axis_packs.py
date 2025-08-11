"""Utility to load axis packs from JSON configuration files."""
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from coherence.axis.pack import AxisPack
from coherence.axis.builder import build_axis_pack_from_vectors
from coherence.encoders.text_sbert import get_default_encoder

def load_axis_configs(config_dir: Path) -> Dict[str, dict]:
    """Load all axis JSON configs from a directory."""
    configs = {}
    for config_file in config_dir.glob("*.json"):
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
            configs[config["name"]] = config
    return configs

def create_axis_pack(config: dict, encoder) -> AxisPack:
    """Create an AxisPack from a single axis config."""
    # Encode the positive and negative examples
    pos_texts = config.get("max_examples", [])
    neg_texts = config.get("min_examples", [])
    
    if not pos_texts or not neg_texts:
        raise ValueError(f"Axis {config.get('name', 'unnamed')} is missing positive or negative examples")
    
    # Encode the examples
    pos_embs = [encoder.encode([text])[0] for text in pos_texts]
    neg_embs = [encoder.encode([text])[0] for text in neg_texts]
    
    # Create the axis pack with the encoded vectors
    axis_name = config.get("name", "unnamed_axis")
    axis_pack = build_axis_pack_from_vectors(
        {axis_name: (pos_embs, neg_embs)},
        weights_init=[config.get("weight", 1.0)]
    )
    
    # Add metadata
    axis_pack.meta.update({
        "description": config.get("description", ""),
        "plain_language_ontology": config.get("plain_language_ontology", ""),
        "plain_language_sought": config.get("plain_language_sought", ""),
        "plain_language_not_sought": config.get("plain_language_not_sought", ""),
        "inclusive_mode": config.get("inclusive_mode", False)
    })
    
    return axis_pack

def load_axis_packs(config_dir: str, device: str = "auto") -> Dict[str, AxisPack]:
    """Load all axis packs from JSON configs in a directory."""
    config_dir = Path(config_dir)
    if not config_dir.exists():
        raise FileNotFoundError(f"Config directory not found: {config_dir}")
    
    # Initialize the encoder
    encoder = get_default_encoder(device=device, normalize_input=True)
    
    # Load all configs and create axis packs
    configs = load_axis_configs(config_dir)
    axis_packs = {}
    
    for name, config in configs.items():
        try:
            axis_pack = create_axis_pack(config, encoder)
            axis_packs[name] = axis_pack
            print(f"✅ Loaded axis: {name} (dimensions: {axis_pack.Q.shape[0]}D -> {axis_pack.Q.shape[1]} axes)")
        except Exception as e:
            print(f"❌ Failed to load axis {name}: {str(e)}")
    
    return axis_packs

if __name__ == "__main__":
    # Example usage
    config_dir = Path(__file__).parent.parent / "Embeddings" / "default embeddings"
    try:
        packs = load_axis_packs(config_dir)
        print(f"\nSuccessfully loaded {len(packs)} axis packs")
    except Exception as e:
        print(f"Error loading axis packs: {str(e)}")
        raise

from __future__ import annotations

import os
import yaml
from pathlib import Path


def project_root() -> Path:
    """Return repository root path based on this file location."""
    return Path(__file__).resolve().parents[3]


def load_app_config() -> dict:
    """Load application configuration from `configs/app.yaml`.

    Returns
    - dict: parsed YAML configuration
    """
    cfg_path = project_root() / "configs" / "app.yaml"
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

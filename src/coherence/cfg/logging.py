from __future__ import annotations

import logging.config
from pathlib import Path
import yaml

from .loader import project_root


def configure_logging() -> None:
    """Configure logging from `configs/logging.yaml`.

    Idempotent; safe to call multiple times.
    """
    cfg_path = project_root() / "configs" / "logging.yaml"
    if cfg_path.exists():
        with cfg_path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=logging.INFO)

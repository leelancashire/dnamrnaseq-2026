"""Configuration loader for dnamrnaseq2026.

Reads config.yaml from the repo root. Falls back to config.yaml.example
if config.yaml is not found (CI / first-clone scenario).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Repo root = two levels up from this file (src/dnamrnaseq2026/data/config.py)
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def _find_config() -> Path:
    """Return the path to config.yaml, falling back to config.yaml.example."""
    primary = _REPO_ROOT / "config.yaml"
    fallback = _REPO_ROOT / "config.yaml.example"
    if primary.exists():
        return primary
    if fallback.exists():
        logger.warning(
            "config.yaml not found; using config.yaml.example. "
            "Run: cp config.yaml.example config.yaml"
        )
        return fallback
    raise FileNotFoundError(
        f"Neither config.yaml nor config.yaml.example found at {_REPO_ROOT}. "
        "Re-clone the repository."
    )


def load_config() -> dict[str, Any]:
    """Load and return the full config dict."""
    config_path = _find_config()
    with config_path.open() as fh:
        cfg: dict[str, Any] = yaml.safe_load(fh)
    return cfg


def get_emory_dnam_dir() -> Path:
    """Return the Emory DNAm data directory as a Path."""
    cfg = load_config()
    return Path(cfg["data"]["emory_dnam_dir"])


def get_emory_mmvae_dir() -> Path:
    """Return the Emory mmVAE supplementary data directory as a Path."""
    cfg = load_config()
    return Path(cfg["data"]["emory_mmvae_dir"])

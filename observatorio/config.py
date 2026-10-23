"""Paths, model ids and run-directory helpers shared by every stage."""

from __future__ import annotations

import json
import os
from pathlib import Path

OBS_ROOT = Path(os.environ.get("OBS_ROOT", Path.home() / ".observatorio"))
DATA = Path(os.environ.get("OBS_DATA", OBS_ROOT / "data"))
RUNS = DATA / "runs"

DATASET = "allenai/WildChat-1M"
LLM_MODEL = "Qwen/Qwen3.8-27B"
EMBED_MODEL = "Qwen/Qwen3-Embedding-8B"


def run_dir(run_id: str) -> Path:
    d = RUNS / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_config(run_id: str, stage: str, params: dict) -> None:
    """Record each stage's parameters in runs/<id>/config.json so a result is reproducible."""
    path = run_dir(run_id) / "config.json"
    cfg = json.loads(path.read_text()) if path.exists() else {}
    cfg[stage] = params
    path.write_text(json.dumps(cfg, indent=2, default=str))

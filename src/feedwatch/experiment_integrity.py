"""Immutable experiment input seals, checked before resuming work or opening a test."""

import json
from pathlib import Path

from .data import file_hash, utc_now, write_json


def seal_inputs(path: Path, inputs: dict) -> dict:
    if path.exists():
        sealed = json.loads(path.read_text(encoding="utf-8"))
        if sealed["inputs"] != inputs:
            raise ValueError(f"Frozen experiment inputs changed: {path.name}")
        return sealed
    sealed = {"sealed_at_utc": utc_now(), "inputs": inputs}
    write_json(path, sealed)
    return sealed


def verify_checkpoint_run(path: Path, result: dict, bundle: dict, expected: dict,
                          seed: int, heads: dict, configuration: dict) -> None:
    if result["seed"] != seed or result["training_inputs"] != expected:
        raise ValueError("Saved seed result belongs to different training inputs")
    if file_hash(path) != result["checkpoint_sha256"]:
        raise ValueError("Saved checkpoint changed after its evaluation")
    if (bundle["seed"] != seed or bundle["kind"] != "gru" or bundle["head_config"] != heads
            or bundle["training_config"] != configuration or bundle["provenance"] != expected
            or bundle["selected_epoch"] != result["selected_epoch"]):
        raise ValueError("Checkpoint configuration/provenance differs from its evaluated run")

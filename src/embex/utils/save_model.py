"""Model-agnostic safetensors and Hugging Face repository export helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Mapping

import numpy as np


StateDictFactory = Callable[[object], Mapping[str, np.ndarray]]


def save_model(
    model: object, output_path: str | Path, state_dict_factory: StateDictFactory
) -> Path:
    """Save model weights with a model-specific state-dictionary adapter."""
    try:
        from safetensors.numpy import save_file
    except ImportError as error:
        raise ImportError("Install `embex[hf]` to save safetensors checkpoints.") from error

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    weights = {
        name: np.ascontiguousarray(value)
        for name, value in state_dict_factory(model).items()
    }
    save_file(weights, str(output_path))
    return output_path


def export_huggingface_model(
    model: object,
    output_dir: str | Path,
    state_dict_factory: StateDictFactory,
    *,
    repo_id: str,
) -> Path:
    """Download a HF model repository and replace its ``model.safetensors`` weights."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise ImportError("Install `embex[hf]` to export Hugging Face repositories.") from error

    output_dir = Path(output_dir)
    snapshot_download(
        repo_id=repo_id,
        local_dir=output_dir,
        ignore_patterns=["*.bin", "*.h5", "*.msgpack"],
    )
    return save_model(model, output_dir / "model.safetensors", state_dict_factory)

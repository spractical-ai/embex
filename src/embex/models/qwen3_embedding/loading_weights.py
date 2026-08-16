"""Hugging Face safetensors adapters for :mod:`embex.models.qwen3_embedding`."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from embex.models.qwen3_embedding.modeling import Qwen3Embedding


def _put_like(weights: np.ndarray, variable_state: Any) -> jax.Array:
    """Convert a NumPy weight to FP32 and retain a variable's existing sharding."""
    array = jnp.asarray(weights, dtype=jnp.float32)
    sharding = getattr(variable_state, "sharding", None)
    return jax.device_put(array, sharding) if sharding is not None else array


def _require_weights(weights: dict[str, np.ndarray], *names: str) -> None:
    missing = [name for name in names if name not in weights]
    if missing:
        raise KeyError(f"Checkpoint is missing required Qwen3 weights: {missing}")


def update_qwen3_embedding_weights(
    weights: dict[str, np.ndarray], model: Qwen3Embedding
) -> None:
    """Load a HF Qwen3 state dictionary into an existing NNX model in place."""
    state = nnx.state(model)
    required = ["norm.weight", "embed_tokens.weight"]
    for layer_index in range(len(state.layers)):
        prefix = f"layers.{layer_index}"
        required.extend(
            [
                f"{prefix}.input_layernorm.weight",
                f"{prefix}.post_attention_layernorm.weight",
                f"{prefix}.mlp.gate_proj.weight",
                f"{prefix}.mlp.up_proj.weight",
                f"{prefix}.mlp.down_proj.weight",
                f"{prefix}.self_attn.q_proj.weight",
                f"{prefix}.self_attn.k_proj.weight",
                f"{prefix}.self_attn.v_proj.weight",
                f"{prefix}.self_attn.o_proj.weight",
                f"{prefix}.self_attn.q_norm.weight",
                f"{prefix}.self_attn.k_norm.weight",
            ]
        )
    _require_weights(weights, *required)

    state.norm.scale = nnx.Param(_put_like(weights["norm.weight"], state.norm.scale))
    state.embed_tokens.embedding = nnx.Param(
        _put_like(weights["embed_tokens.weight"], state.embed_tokens.embedding)
    )
    for layer_index in range(len(state.layers)):
        layer = state.layers[layer_index]
        prefix = f"layers.{layer_index}"
        layer.input_layernorm.scale = nnx.Param(
            _put_like(weights[f"{prefix}.input_layernorm.weight"], layer.input_layernorm.scale)
        )
        layer.post_attention_layernorm.scale = nnx.Param(
            _put_like(
                weights[f"{prefix}.post_attention_layernorm.weight"],
                layer.post_attention_layernorm.scale,
            )
        )
        for name in ("gate_proj", "up_proj", "down_proj"):
            variable = getattr(layer.mlp, name).kernel
            getattr(layer.mlp, name).kernel = nnx.Param(
                _put_like(weights[f"{prefix}.mlp.{name}.weight"].T, variable)
            )
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            variable = getattr(layer.self_attn, name).kernel
            getattr(layer.self_attn, name).kernel = nnx.Param(
                _put_like(weights[f"{prefix}.self_attn.{name}.weight"].T, variable)
            )
        layer.self_attn.q_norm.scale = nnx.Param(
            _put_like(weights[f"{prefix}.self_attn.q_norm.weight"], layer.self_attn.q_norm.scale)
        )
        layer.self_attn.k_norm.scale = nnx.Param(
            _put_like(weights[f"{prefix}.self_attn.k_norm.weight"], layer.self_attn.k_norm.scale)
        )
    nnx.update(model, state)


def load_qwen3_embedding_weights(
    model: Qwen3Embedding, checkpoint_path: str | Path
) -> None:
    """Load an HF ``model.safetensors`` checkpoint into ``model``."""
    try:
        from safetensors.numpy import load_file
    except ImportError as error:
        raise ImportError("Install `embex[hf]` to load safetensors checkpoints.") from error
    update_qwen3_embedding_weights(load_file(str(checkpoint_path)), model)


def _array(value: Any) -> np.ndarray:
    """Materialize an NNX variable as a FP32 NumPy array."""
    if hasattr(value, "get_value"):
        value = value.get_value()
    elif hasattr(value, "value"):
        value = value.value
    return np.asarray(value, dtype=np.float32)


def qwen3_embedding_hf_state_dict(model: Qwen3Embedding) -> dict[str, np.ndarray]:
    """Return Qwen3 weights in the Hugging Face safetensors naming and layout."""
    state = nnx.state(model)
    weights = {
        "norm.weight": _array(state.norm.scale),
        "embed_tokens.weight": _array(state.embed_tokens.embedding),
    }
    for layer_index in range(len(state.layers)):
        layer = state.layers[layer_index]
        prefix = f"layers.{layer_index}"
        weights[f"{prefix}.input_layernorm.weight"] = _array(layer.input_layernorm.scale)
        weights[f"{prefix}.post_attention_layernorm.weight"] = _array(
            layer.post_attention_layernorm.scale
        )
        weights[f"{prefix}.self_attn.q_norm.weight"] = _array(layer.self_attn.q_norm.scale)
        weights[f"{prefix}.self_attn.k_norm.weight"] = _array(layer.self_attn.k_norm.scale)
        for name in ("gate_proj", "up_proj", "down_proj"):
            weights[f"{prefix}.mlp.{name}.weight"] = _array(
                getattr(layer.mlp, name).kernel
            ).T
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            weights[f"{prefix}.self_attn.{name}.weight"] = _array(
                getattr(layer.self_attn, name).kernel
            ).T
    return weights

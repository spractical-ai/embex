"""Hugging Face safetensors adapters for :mod:`embex.models.xlm_roberta`."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from embex.models.xlm_roberta.modeling import XLMRobertaEmbedding


_CHECKPOINT_PREFIXES = ("", "roberta.", "xlm_roberta.", "model.")
_MLM_CHECKPOINT_PREFIXES = ("", "model.")


def _put_like(weights: np.ndarray, variable_state: Any) -> jax.Array:
    """Convert a NumPy weight to FP32 and retain existing variable sharding."""
    array = jnp.asarray(weights, dtype=jnp.float32)
    sharding = getattr(variable_state, "sharding", None)
    return jax.device_put(array, sharding) if sharding is not None else array


def _resolve_weights(
    weights: dict[str, np.ndarray], names: list[str]
) -> dict[str, np.ndarray]:
    """Resolve bare XLM-R keys and common Hugging Face backbone prefixes."""
    for prefix in _CHECKPOINT_PREFIXES:
        if all(f"{prefix}{name}" in weights for name in names):
            return {name: weights[f"{prefix}{name}"] for name in names}
    missing = [
        name
        for name in names
        if not any(f"{prefix}{name}" in weights for prefix in _CHECKPOINT_PREFIXES)
    ]
    raise KeyError(f"Checkpoint is missing required XLM-RoBERTa weights: {missing}")


def _required_weight_names(num_layers: int) -> list[str]:
    names = [
        "embeddings.word_embeddings.weight",
        "embeddings.position_embeddings.weight",
        "embeddings.token_type_embeddings.weight",
        "embeddings.LayerNorm.weight",
        "embeddings.LayerNorm.bias",
    ]
    for layer_index in range(num_layers):
        prefix = f"encoder.layer.{layer_index}"
        names.extend(
            [
                f"{prefix}.attention.self.query.weight",
                f"{prefix}.attention.self.query.bias",
                f"{prefix}.attention.self.key.weight",
                f"{prefix}.attention.self.key.bias",
                f"{prefix}.attention.self.value.weight",
                f"{prefix}.attention.self.value.bias",
                f"{prefix}.attention.output.dense.weight",
                f"{prefix}.attention.output.dense.bias",
                f"{prefix}.attention.output.LayerNorm.weight",
                f"{prefix}.attention.output.LayerNorm.bias",
                f"{prefix}.intermediate.dense.weight",
                f"{prefix}.intermediate.dense.bias",
                f"{prefix}.output.dense.weight",
                f"{prefix}.output.dense.bias",
                f"{prefix}.output.LayerNorm.weight",
                f"{prefix}.output.LayerNorm.bias",
            ]
        )
    return names


def _resolve_optional_mlm_weights(
    weights: dict[str, np.ndarray],
) -> dict[str, np.ndarray] | None:
    names = [
        "lm_head.dense.weight",
        "lm_head.dense.bias",
        "lm_head.layer_norm.weight",
        "lm_head.layer_norm.bias",
    ]
    for prefix in _MLM_CHECKPOINT_PREFIXES:
        bias_name = next(
            (
                name
                for name in ("lm_head.bias", "lm_head.decoder.bias")
                if f"{prefix}{name}" in weights
            ),
            None,
        )
        has_transform = all(f"{prefix}{name}" in weights for name in names)
        if bias_name is not None and has_transform:
            resolved = {name: weights[f"{prefix}{name}"] for name in names}
            resolved["lm_head.bias"] = weights[f"{prefix}{bias_name}"]
            return resolved

    head_keys = [name for name in weights if "lm_head." in name]
    if not head_keys:
        return None
    raise KeyError(
        "Checkpoint contains an incomplete or unsupported XLM-RoBERTa MLM head."
    )


def _set_linear(linear: Any, weight: np.ndarray, bias: np.ndarray) -> None:
    linear.kernel = nnx.Param(_put_like(weight.T, linear.kernel))
    linear.bias = nnx.Param(_put_like(bias, linear.bias))


def _set_layer_norm(layer_norm: Any, weight: np.ndarray, bias: np.ndarray) -> None:
    layer_norm.scale = nnx.Param(_put_like(weight, layer_norm.scale))
    layer_norm.bias = nnx.Param(_put_like(bias, layer_norm.bias))


def update_xlm_roberta_weights(
    weights: dict[str, np.ndarray], model: XLMRobertaEmbedding
) -> None:
    """Load an XLM-R backbone and, when present, its MLM head in place."""
    state = nnx.state(model)
    resolved = _resolve_weights(weights, _required_weight_names(len(state.layers)))
    mlm_weights = _resolve_optional_mlm_weights(weights)
    embeddings = state.embeddings
    embeddings.word_embeddings.embedding = nnx.Param(
        _put_like(
            resolved["embeddings.word_embeddings.weight"],
            embeddings.word_embeddings.embedding,
        )
    )
    embeddings.position_embeddings.embedding = nnx.Param(
        _put_like(
            resolved["embeddings.position_embeddings.weight"],
            embeddings.position_embeddings.embedding,
        )
    )
    embeddings.token_type_embeddings.embedding = nnx.Param(
        _put_like(
            resolved["embeddings.token_type_embeddings.weight"],
            embeddings.token_type_embeddings.embedding,
        )
    )
    _set_layer_norm(
        embeddings.layer_norm,
        resolved["embeddings.LayerNorm.weight"],
        resolved["embeddings.LayerNorm.bias"],
    )

    for layer_index in range(len(state.layers)):
        layer = state.layers[layer_index]
        prefix = f"encoder.layer.{layer_index}"
        for name in ("query", "key", "value"):
            _set_linear(
                getattr(layer.attention, name),
                resolved[f"{prefix}.attention.self.{name}.weight"],
                resolved[f"{prefix}.attention.self.{name}.bias"],
            )
        _set_linear(
            layer.attention.dense,
            resolved[f"{prefix}.attention.output.dense.weight"],
            resolved[f"{prefix}.attention.output.dense.bias"],
        )
        _set_layer_norm(
            layer.attention.layer_norm,
            resolved[f"{prefix}.attention.output.LayerNorm.weight"],
            resolved[f"{prefix}.attention.output.LayerNorm.bias"],
        )
        _set_linear(
            layer.intermediate,
            resolved[f"{prefix}.intermediate.dense.weight"],
            resolved[f"{prefix}.intermediate.dense.bias"],
        )
        _set_linear(
            layer.output,
            resolved[f"{prefix}.output.dense.weight"],
            resolved[f"{prefix}.output.dense.bias"],
        )
        _set_layer_norm(
            layer.output_layer_norm,
            resolved[f"{prefix}.output.LayerNorm.weight"],
            resolved[f"{prefix}.output.LayerNorm.bias"],
        )
    if mlm_weights is not None:
        _set_linear(
            state.lm_head.dense,
            mlm_weights["lm_head.dense.weight"],
            mlm_weights["lm_head.dense.bias"],
        )
        _set_layer_norm(
            state.lm_head.layer_norm,
            mlm_weights["lm_head.layer_norm.weight"],
            mlm_weights["lm_head.layer_norm.bias"],
        )
        state.lm_head.bias = nnx.Param(
            _put_like(mlm_weights["lm_head.bias"], state.lm_head.bias)
        )
    nnx.update(model, state)


def load_xlm_roberta_weights(
    model: XLMRobertaEmbedding, checkpoint_path: str | Path
) -> None:
    """Load a Hugging Face ``model.safetensors`` checkpoint into ``model``."""
    try:
        from safetensors.numpy import load_file
    except ImportError as error:
        raise ImportError("Install `safetensors` to load checkpoints.") from error
    update_xlm_roberta_weights(load_file(str(checkpoint_path)), model)


def _array(value: Any) -> np.ndarray:
    """Materialize an NNX variable as a FP32 NumPy array."""
    if hasattr(value, "get_value"):
        value = value.get_value()
    elif hasattr(value, "value"):
        value = value.value
    return np.asarray(value, dtype=np.float32)


def _export_linear(
    weights: dict[str, np.ndarray], prefix: str, linear: Any
) -> None:
    weights[f"{prefix}.weight"] = np.ascontiguousarray(_array(linear.kernel).T)
    weights[f"{prefix}.bias"] = _array(linear.bias)


def _export_layer_norm(
    weights: dict[str, np.ndarray], prefix: str, layer_norm: Any
) -> None:
    weights[f"{prefix}.weight"] = _array(layer_norm.scale)
    weights[f"{prefix}.bias"] = _array(layer_norm.bias)


def xlm_roberta_hf_state_dict(
    model: XLMRobertaEmbedding,
) -> dict[str, np.ndarray]:
    """Return backbone weights in Hugging Face XLM-RoBERTa naming and layout."""
    state = nnx.state(model)
    embeddings = state.embeddings
    weights = {
        "embeddings.word_embeddings.weight": _array(
            embeddings.word_embeddings.embedding
        ),
        "embeddings.position_embeddings.weight": _array(
            embeddings.position_embeddings.embedding
        ),
        "embeddings.token_type_embeddings.weight": _array(
            embeddings.token_type_embeddings.embedding
        ),
    }
    _export_layer_norm(weights, "embeddings.LayerNorm", embeddings.layer_norm)
    for layer_index in range(len(state.layers)):
        layer = state.layers[layer_index]
        prefix = f"encoder.layer.{layer_index}"
        for name in ("query", "key", "value"):
            _export_linear(
                weights,
                f"{prefix}.attention.self.{name}",
                getattr(layer.attention, name),
            )
        _export_linear(
            weights, f"{prefix}.attention.output.dense", layer.attention.dense
        )
        _export_layer_norm(
            weights,
            f"{prefix}.attention.output.LayerNorm",
            layer.attention.layer_norm,
        )
        _export_linear(weights, f"{prefix}.intermediate.dense", layer.intermediate)
        _export_linear(weights, f"{prefix}.output.dense", layer.output)
        _export_layer_norm(
            weights, f"{prefix}.output.LayerNorm", layer.output_layer_norm
        )
    return weights


def xlm_roberta_mlm_hf_state_dict(
    model: XLMRobertaEmbedding,
) -> dict[str, np.ndarray]:
    """Return a Hugging Face XLM-RoBERTa masked-LM state dictionary."""
    weights = {
        f"roberta.{name}": value
        for name, value in xlm_roberta_hf_state_dict(model).items()
    }
    state = nnx.state(model)
    _export_linear(weights, "lm_head.dense", state.lm_head.dense)
    _export_layer_norm(weights, "lm_head.layer_norm", state.lm_head.layer_norm)
    weights["lm_head.bias"] = _array(state.lm_head.bias)
    return weights

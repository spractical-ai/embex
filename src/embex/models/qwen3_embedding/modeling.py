"""Flax NNX implementation of the Qwen3 embedding encoder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh

from embex.utils.distributed import create_fsdp_mesh, mesh_context


@dataclass
class Qwen3EmbeddingConfig:
    """Architecture and precision settings for a Qwen3 embedding model."""

    vocab_size: int = 151669
    hidden_size: int = 1024
    head_dim: int = 128
    intermediate_size: int = 3072
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    param_dtype: Any = jnp.float32
    dtype: Any = jnp.bfloat16

    def __post_init__(self) -> None:
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads."
            )

    @classmethod
    def from_preset(cls, model_name: str, **overrides: Any) -> "Qwen3EmbeddingConfig":
        """Return a Qwen3 architecture preset, optionally applying overrides."""
        name = model_name.lower()
        if "0.6b" in name:
            config = cls(
                vocab_size=151669,
                hidden_size=1024,
                intermediate_size=3072,
                num_hidden_layers=28,
                num_attention_heads=16,
                num_key_value_heads=8,
            )
        elif "4b" in name:
            config = cls(
                vocab_size=151665,
                hidden_size=2560,
                intermediate_size=9728,
                num_hidden_layers=36,
                num_attention_heads=32,
                num_key_value_heads=8,
            )
        elif "8b" in name:
            config = cls(
                vocab_size=151665,
                hidden_size=4096,
                intermediate_size=12288,
                num_hidden_layers=36,
                num_attention_heads=32,
                num_key_value_heads=8,
            )
        else:
            raise ValueError("Use a Qwen3 preset containing '0.6B', '4B', or '8B'.")

        for key, value in overrides.items():
            if not hasattr(config, key):
                raise KeyError(f"Invalid Qwen3EmbeddingConfig attribute: {key}")
            setattr(config, key, value)
        config.__post_init__()
        return config


def compute_rope_cos_sin(
    seq_len: int, head_dim: int, theta: float = 1_000_000.0
) -> tuple[jax.Array, jax.Array]:
    """Compute Qwen RoPE cosine and sine tensors with broadcastable dimensions."""
    inv_freq = 1.0 / (
        theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim)
    )
    freqs = jnp.outer(jnp.arange(seq_len, dtype=jnp.float32), inv_freq)
    embeddings = jnp.concatenate((freqs, freqs), axis=-1)
    return jnp.cos(embeddings)[None, :, None, :], jnp.sin(embeddings)[None, :, None, :]


def apply_rotary_emb(
    queries: jax.Array, keys: jax.Array, cos: jax.Array, sin: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Apply Qwen's rotate-half RoPE formulation."""

    def rotate_half(values: jax.Array) -> jax.Array:
        half_dim = values.shape[-1] // 2
        return jnp.concatenate((-values[..., half_dim:], values[..., :half_dim]), axis=-1)

    return (
        (queries * cos) + (rotate_half(queries) * sin),
        (keys * cos) + (rotate_half(keys) * sin),
    )


def _kernel_init(partition_axis: str | None, axes: tuple[None | str, None | str]):
    initializer = nnx.initializers.lecun_normal()
    if partition_axis is None:
        return initializer
    partition_spec = tuple(partition_axis if axis == "fsdp" else axis for axis in axes)
    return nnx.with_partitioning(initializer, partition_spec)


class SwiGLUMLP(nnx.Module):
    def __init__(
        self,
        config: Qwen3EmbeddingConfig,
        rngs: nnx.Rngs,
        partition_axis: str | None,
    ) -> None:
        kwargs = {
            "use_bias": False,
            "dtype": config.dtype,
            "param_dtype": config.param_dtype,
            "rngs": rngs,
        }
        self.gate_proj = nnx.Linear(
            config.hidden_size,
            config.intermediate_size,
            kernel_init=_kernel_init(partition_axis, (None, "fsdp")),
            **kwargs,
        )
        self.up_proj = nnx.Linear(
            config.hidden_size,
            config.intermediate_size,
            kernel_init=_kernel_init(partition_axis, (None, "fsdp")),
            **kwargs,
        )
        self.down_proj = nnx.Linear(
            config.intermediate_size,
            config.hidden_size,
            kernel_init=_kernel_init(partition_axis, ("fsdp", None)),
            **kwargs,
        )

    def __call__(self, hidden_states: jax.Array) -> jax.Array:
        return self.down_proj(
            nnx.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        )


class GroupQueryAttention(nnx.Module):
    def __init__(
        self,
        config: Qwen3EmbeddingConfig,
        rngs: nnx.Rngs,
        partition_axis: str | None,
    ) -> None:
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        kwargs = {
            "use_bias": False,
            "dtype": config.dtype,
            "param_dtype": config.param_dtype,
            "rngs": rngs,
        }
        self.q_proj = nnx.Linear(
            config.hidden_size,
            self.num_heads * self.head_dim,
            kernel_init=_kernel_init(partition_axis, (None, "fsdp")),
            **kwargs,
        )
        self.k_proj = nnx.Linear(
            config.hidden_size,
            self.num_kv_heads * self.head_dim,
            kernel_init=_kernel_init(partition_axis, (None, "fsdp")),
            **kwargs,
        )
        self.v_proj = nnx.Linear(
            config.hidden_size,
            self.num_kv_heads * self.head_dim,
            kernel_init=_kernel_init(partition_axis, (None, "fsdp")),
            **kwargs,
        )
        self.o_proj = nnx.Linear(
            self.num_heads * self.head_dim,
            config.hidden_size,
            kernel_init=_kernel_init(partition_axis, ("fsdp", None)),
            **kwargs,
        )
        norm_kwargs = {
            "epsilon": config.rms_norm_eps,
            "dtype": jnp.float32,
            "param_dtype": jnp.float32,
            "rngs": rngs,
        }
        self.q_norm = nnx.RMSNorm(self.head_dim, **norm_kwargs)
        self.k_norm = nnx.RMSNorm(self.head_dim, **norm_kwargs)

    def __call__(
        self,
        hidden_states: jax.Array,
        cos: jax.Array,
        sin: jax.Array,
        mask: jax.Array | None,
    ) -> jax.Array:
        batch_size, seq_len, _ = hidden_states.shape
        queries = self.q_proj(hidden_states).reshape(
            batch_size, seq_len, self.num_heads, self.head_dim
        )
        keys = self.k_proj(hidden_states).reshape(
            batch_size, seq_len, self.num_kv_heads, self.head_dim
        )
        values = self.v_proj(hidden_states).reshape(
            batch_size, seq_len, self.num_kv_heads, self.head_dim
        )

        queries = self.q_norm(queries.astype(jnp.float32)).astype(hidden_states.dtype)
        keys = self.k_norm(keys.astype(jnp.float32)).astype(hidden_states.dtype)
        queries, keys = apply_rotary_emb(
            queries.astype(jnp.float32), keys.astype(jnp.float32), cos, sin
        )
        queries = queries.astype(hidden_states.dtype)
        keys = keys.astype(hidden_states.dtype)

        keys = jnp.repeat(keys, self.num_queries_per_kv, axis=2)
        values = jnp.repeat(values, self.num_queries_per_kv, axis=2)
        queries = queries.transpose(0, 2, 1, 3)
        keys = keys.transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)

        scale = 1.0 / jnp.sqrt(self.head_dim)
        scores = jnp.matmul(queries, keys.transpose(0, 1, 3, 2)) * scale
        if mask is not None:
            scores = scores + mask.astype(scores.dtype)
        probabilities = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(
            hidden_states.dtype
        )
        context = jnp.matmul(probabilities, values)
        context = context.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        return self.o_proj(context)


class QwenBlock(nnx.Module):
    def __init__(
        self,
        config: Qwen3EmbeddingConfig,
        rngs: nnx.Rngs,
        partition_axis: str | None,
    ) -> None:
        norm_kwargs = {
            "epsilon": config.rms_norm_eps,
            "dtype": jnp.float32,
            "param_dtype": jnp.float32,
            "rngs": rngs,
        }
        self.input_layernorm = nnx.RMSNorm(config.hidden_size, **norm_kwargs)
        self.self_attn = GroupQueryAttention(config, rngs, partition_axis)
        self.post_attention_layernorm = nnx.RMSNorm(config.hidden_size, **norm_kwargs)
        self.mlp = SwiGLUMLP(config, rngs, partition_axis)

    @nnx.remat
    def __call__(
        self,
        hidden_states: jax.Array,
        cos: jax.Array,
        sin: jax.Array,
        mask: jax.Array | None,
    ) -> jax.Array:
        normed = self.input_layernorm(hidden_states.astype(jnp.float32)).astype(
            hidden_states.dtype
        )
        hidden_states = hidden_states + self.self_attn(normed, cos, sin, mask)
        normed = self.post_attention_layernorm(hidden_states.astype(jnp.float32)).astype(
            hidden_states.dtype
        )
        return hidden_states + self.mlp(normed)


class Qwen3Embedding(nnx.Module):
    """Decoder-only Qwen3 encoder that returns a pooled sequence embedding."""

    def __init__(
        self,
        config: Qwen3EmbeddingConfig,
        rngs: nnx.Rngs,
        partition_axis: str | None = None,
    ) -> None:
        if jax.device_count() > 1 and partition_axis is None:
            raise ValueError(
                "Multi-device construction requires an active mesh and a partition axis. "
                "Use create_qwen3_embedding(), or enter `with jax.set_mesh(mesh):` "
                "and pass that mesh axis as partition_axis."
            )
        self.config = config
        self.embed_tokens = nnx.Embed(
            config.vocab_size,
            config.hidden_size,
            param_dtype=config.param_dtype,
            dtype=config.dtype,
            embedding_init=_kernel_init(partition_axis, (None, "fsdp")),
            rngs=rngs,
        )
        self.layers = nnx.List(
            QwenBlock(config, rngs, partition_axis)
            for _ in range(config.num_hidden_layers)
        )
        self.norm = nnx.RMSNorm(config.hidden_size, epsilon=config.rms_norm_eps, rngs=rngs)

    def __call__(
        self, input_ids: jax.Array, attention_mask: jax.Array | None = None
    ) -> jax.Array:
        batch_size, seq_len = input_ids.shape
        hidden_states = self.embed_tokens(input_ids)
        causal_mask = jnp.tril(jnp.ones((seq_len, seq_len)))
        if attention_mask is None:
            mask = jnp.where(causal_mask == 0, -1e9, 0.0).reshape(1, 1, seq_len, seq_len)
        else:
            combined_mask = causal_mask * attention_mask[:, None, None, :]
            mask = jnp.where(combined_mask == 0, -1e9, 0.0)

        cos, sin = compute_rope_cos_sin(
            seq_len, self.config.head_dim, self.config.rope_theta
        )
        for layer in self.layers:
            hidden_states = layer(hidden_states, cos, sin, mask)
        hidden_states = self.norm(hidden_states)

        if attention_mask is None:
            return hidden_states[:, -1, :]

        is_left_padded = jnp.sum(attention_mask[:, -1]) == batch_size
        sequence_lengths = jnp.sum(attention_mask, axis=1) - 1
        target_indices = jnp.where(is_left_padded, seq_len - 1, sequence_lengths)
        return hidden_states[jnp.arange(batch_size), target_indices, :]


def create_qwen3_embedding(
    config: Qwen3EmbeddingConfig,
    *,
    seed: int = 0,
    mesh: Mesh | None = None,
    partition_axis: str | None = None,
) -> tuple[Qwen3Embedding, Mesh | None]:
    """Create a model, using one-axis FSDP only when multiple devices are available.

    Pass a mesh to control placement explicitly.  When no mesh is given, EmbeX creates
    a single-axis FSDP mesh for multi-device JAX runtimes and uses ordinary initializers
    for a single device.
    """
    if mesh is None and jax.device_count() > 1:
        mesh = create_fsdp_mesh()
    if mesh is not None and partition_axis is None:
        partition_axis = str(mesh.axis_names[0])
    if mesh is None:
        partition_axis = None

    with mesh_context(mesh):
        model = Qwen3Embedding(
            config,
            rngs=nnx.Rngs(params=seed),
            partition_axis=partition_axis,
        )
    return model, mesh

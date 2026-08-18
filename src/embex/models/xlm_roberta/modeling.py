"""Flax NNX implementation of a pooled XLM-RoBERTa encoder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh

from embex.utils.distributed import create_fsdp_mesh, mesh_context


@dataclass
class XLMRobertaConfig:
    """Architecture, precision, and pooling settings for XLM-RoBERTa."""

    vocab_size: int = 250002
    hidden_size: int = 768
    intermediate_size: int = 3072
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    max_position_embeddings: int = 514
    type_vocab_size: int = 1
    pad_token_id: int = 1
    layer_norm_eps: float = 1e-5
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    initializer_range: float = 0.02
    pooling: str = "mean"
    param_dtype: Any = jnp.float32
    dtype: Any = jnp.bfloat16

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads.")
        if self.pooling not in {"mean", "cls"}:
            raise ValueError("pooling must be either 'mean' or 'cls'.")
        for name in ("hidden_dropout_prob", "attention_probs_dropout_prob"):
            value = getattr(self, name)
            if not 0.0 <= value < 1.0:
                raise ValueError(f"{name} must be in the range [0, 1).")

    @classmethod
    def from_preset(cls, model_name: str, **overrides: Any) -> "XLMRobertaConfig":
        """Return an XLM-RoBERTa base or large preset with optional overrides."""
        name = model_name.lower().replace("_", "-")
        if "large" in name:
            config = cls(
                hidden_size=1024,
                intermediate_size=4096,
                num_hidden_layers=24,
                num_attention_heads=16,
            )
        elif "base" in name:
            config = cls()
        else:
            raise ValueError("Use an XLM-RoBERTa preset containing 'base' or 'large'.")

        for key, value in overrides.items():
            if not hasattr(config, key):
                raise KeyError(f"Invalid XLMRobertaConfig attribute: {key}")
            setattr(config, key, value)
        config.__post_init__()
        return config


def create_position_ids(input_ids: jax.Array, padding_idx: int) -> jax.Array:
    """Create fairseq-style positions while keeping padding at ``padding_idx``."""
    mask = (input_ids != padding_idx).astype(jnp.int32)
    return (jnp.cumsum(mask, axis=1) * mask) + padding_idx


def _kernel_init(
    partition_axis: str | None,
    axes: tuple[None | str, None | str],
    initializer_range: float,
):
    initializer = nnx.initializers.normal(stddev=initializer_range)
    if partition_axis is None:
        return initializer
    partition_spec = tuple(partition_axis if axis == "fsdp" else axis for axis in axes)
    return nnx.with_partitioning(initializer, partition_spec)


def _linear(
    in_features: int,
    out_features: int,
    config: XLMRobertaConfig,
    rngs: nnx.Rngs,
    partition_axis: str | None,
    axes: tuple[None | str, None | str],
) -> nnx.Linear:
    return nnx.Linear(
        in_features,
        out_features,
        use_bias=True,
        dtype=config.dtype,
        param_dtype=config.param_dtype,
        kernel_init=_kernel_init(partition_axis, axes, config.initializer_range),
        bias_init=nnx.initializers.zeros_init(),
        rngs=rngs,
    )


def _layer_norm(config: XLMRobertaConfig, rngs: nnx.Rngs) -> nnx.LayerNorm:
    return nnx.LayerNorm(
        config.hidden_size,
        epsilon=config.layer_norm_eps,
        dtype=jnp.float32,
        param_dtype=config.param_dtype,
        rngs=rngs,
    )


class XLMRobertaEmbeddings(nnx.Module):
    def __init__(
        self,
        config: XLMRobertaConfig,
        rngs: nnx.Rngs,
        partition_axis: str | None,
    ) -> None:
        embedding_init = _kernel_init(
            partition_axis, (None, "fsdp"), config.initializer_range
        )
        embedding_kwargs = {
            "param_dtype": config.param_dtype,
            "dtype": config.dtype,
            "embedding_init": embedding_init,
            "rngs": rngs,
        }
        self.word_embeddings = nnx.Embed(
            config.vocab_size, config.hidden_size, **embedding_kwargs
        )
        self.position_embeddings = nnx.Embed(
            config.max_position_embeddings, config.hidden_size, **embedding_kwargs
        )
        self.token_type_embeddings = nnx.Embed(
            config.type_vocab_size, config.hidden_size, **embedding_kwargs
        )
        self.layer_norm = _layer_norm(config, rngs)
        self.dropout = nnx.Dropout(config.hidden_dropout_prob, rngs=rngs)

    def __call__(
        self,
        input_ids: jax.Array,
        position_ids: jax.Array,
        *,
        deterministic: bool,
    ) -> jax.Array:
        token_type_ids = jnp.zeros_like(input_ids)
        hidden_states = (
            self.word_embeddings(input_ids)
            + self.position_embeddings(position_ids)
            + self.token_type_embeddings(token_type_ids)
        )
        hidden_states = self.layer_norm(hidden_states.astype(jnp.float32)).astype(
            self.word_embeddings.dtype
        )
        return self.dropout(hidden_states, deterministic=deterministic)


class XLMRobertaAttention(nnx.Module):
    def __init__(
        self,
        config: XLMRobertaConfig,
        rngs: nnx.Rngs,
        partition_axis: str | None,
    ) -> None:
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        for name in ("query", "key", "value"):
            setattr(
                self,
                name,
                _linear(
                    config.hidden_size,
                    config.hidden_size,
                    config,
                    rngs,
                    partition_axis,
                    (None, "fsdp"),
                ),
            )
        self.dense = _linear(
            config.hidden_size,
            config.hidden_size,
            config,
            rngs,
            partition_axis,
            ("fsdp", None),
        )
        self.layer_norm = _layer_norm(config, rngs)
        self.probability_dropout = nnx.Dropout(
            config.attention_probs_dropout_prob, rngs=rngs
        )
        self.output_dropout = nnx.Dropout(config.hidden_dropout_prob, rngs=rngs)

    def __call__(
        self,
        hidden_states: jax.Array,
        mask: jax.Array | None,
        deterministic: bool,
    ) -> jax.Array:
        batch_size, seq_len, hidden_size = hidden_states.shape

        def split_heads(values: jax.Array) -> jax.Array:
            return values.reshape(
                batch_size, seq_len, self.num_heads, self.head_dim
            ).transpose(0, 2, 1, 3)

        queries = split_heads(self.query(hidden_states)).astype(jnp.float32)
        keys = split_heads(self.key(hidden_states)).astype(jnp.float32)
        values = split_heads(self.value(hidden_states))
        scores = jnp.matmul(queries, keys.transpose(0, 1, 3, 2))
        scores = scores * (self.head_dim**-0.5)
        if mask is not None:
            scores = scores + mask
        probabilities = jax.nn.softmax(scores, axis=-1)
        probabilities = self.probability_dropout(
            probabilities, deterministic=deterministic
        ).astype(values.dtype)
        context = jnp.matmul(probabilities, values)
        context = context.transpose(0, 2, 1, 3).reshape(
            batch_size, seq_len, hidden_size
        )
        context = self.output_dropout(
            self.dense(context), deterministic=deterministic
        )
        return self.layer_norm(
            (context + hidden_states).astype(jnp.float32)
        ).astype(hidden_states.dtype)


class XLMRobertaBlock(nnx.Module):
    def __init__(
        self,
        config: XLMRobertaConfig,
        rngs: nnx.Rngs,
        partition_axis: str | None,
    ) -> None:
        self.attention = XLMRobertaAttention(config, rngs, partition_axis)
        self.intermediate = _linear(
            config.hidden_size,
            config.intermediate_size,
            config,
            rngs,
            partition_axis,
            (None, "fsdp"),
        )
        self.output = _linear(
            config.intermediate_size,
            config.hidden_size,
            config,
            rngs,
            partition_axis,
            ("fsdp", None),
        )
        self.output_layer_norm = _layer_norm(config, rngs)
        self.output_dropout = nnx.Dropout(config.hidden_dropout_prob, rngs=rngs)

    @nnx.remat(static_argnums=3)
    def __call__(
        self,
        hidden_states: jax.Array,
        mask: jax.Array | None,
        deterministic: bool,
    ) -> jax.Array:
        attention_output = self.attention(
            hidden_states, mask, deterministic=deterministic
        )
        intermediate = jax.nn.gelu(
            self.intermediate(attention_output).astype(jnp.float32), approximate=False
        ).astype(attention_output.dtype)
        output = self.output_dropout(
            self.output(intermediate), deterministic=deterministic
        )
        return self.output_layer_norm(
            (output + attention_output).astype(jnp.float32)
        ).astype(attention_output.dtype)


class XLMRobertaMLMHead(nnx.Module):
    """Hugging Face-compatible transform and tied decoder for MLM logits."""

    def __init__(
        self,
        config: XLMRobertaConfig,
        rngs: nnx.Rngs,
        partition_axis: str | None,
    ) -> None:
        self.dense = _linear(
            config.hidden_size,
            config.hidden_size,
            config,
            rngs,
            partition_axis,
            (None, "fsdp"),
        )
        self.layer_norm = _layer_norm(config, rngs)
        self.bias = nnx.Param(jnp.zeros((config.vocab_size,), config.param_dtype))

    def __call__(
        self, hidden_states: jax.Array, word_embeddings: jax.Array
    ) -> jax.Array:
        hidden_states = jax.nn.gelu(
            self.dense(hidden_states).astype(jnp.float32), approximate=False
        )
        hidden_states = self.layer_norm(hidden_states)
        logits = jnp.einsum(
            "bsh,vh->bsv",
            hidden_states.astype(jnp.float32),
            word_embeddings.astype(jnp.float32),
        )
        return logits + self.bias.astype(jnp.float32)


class XLMRobertaEmbedding(nnx.Module):
    """XLM-R encoder supporting pooled embeddings and tied-decoder MLM logits."""

    def __init__(
        self,
        config: XLMRobertaConfig,
        rngs: nnx.Rngs,
        partition_axis: str | None = None,
    ) -> None:
        if jax.device_count() > 1 and partition_axis is None:
            raise ValueError(
                "Multi-device construction requires an active mesh and a "
                "partition axis. "
                "Use create_xlm_roberta(), or enter `with jax.set_mesh(mesh):` "
                "and pass that mesh axis as partition_axis."
            )
        self.config = config
        self.embeddings = XLMRobertaEmbeddings(config, rngs, partition_axis)
        self.layers = nnx.List(
            XLMRobertaBlock(config, rngs, partition_axis)
            for _ in range(config.num_hidden_layers)
        )
        self.lm_head = XLMRobertaMLMHead(config, rngs, partition_axis)

    def encode_tokens(
        self,
        input_ids: jax.Array,
        attention_mask: jax.Array | None = None,
        *,
        deterministic: bool = True,
    ) -> jax.Array:
        """Return contextual token representations before sequence pooling."""
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape (batch, sequence).")
        if attention_mask is not None and attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must have the same shape as input_ids.")
        seq_len = input_ids.shape[1]
        if seq_len + self.config.pad_token_id >= self.config.max_position_embeddings:
            raise ValueError(
                "The input sequence is too long for max_position_embeddings."
            )

        position_ids = create_position_ids(input_ids, self.config.pad_token_id)
        hidden_states = self.embeddings(
            input_ids, position_ids, deterministic=deterministic
        )
        mask = None
        if attention_mask is not None:
            mask = jnp.where(attention_mask[:, None, None, :] != 0, 0.0, -1e9)
        for layer in self.layers:
            hidden_states = layer(hidden_states, mask, deterministic)
        return hidden_states

    def masked_language_model_logits(
        self,
        input_ids: jax.Array,
        attention_mask: jax.Array | None = None,
        *,
        deterministic: bool = True,
    ) -> jax.Array:
        """Return per-token vocabulary logits from the tied XLM-R MLM head."""
        hidden_states = self.encode_tokens(
            input_ids, attention_mask, deterministic=deterministic
        )
        return self.lm_head(
            hidden_states, self.embeddings.word_embeddings.embedding
        )

    def __call__(
        self,
        input_ids: jax.Array,
        attention_mask: jax.Array | None = None,
        *,
        deterministic: bool = True,
    ) -> jax.Array:
        hidden_states = self.encode_tokens(
            input_ids, attention_mask, deterministic=deterministic
        )
        if self.config.pooling == "cls":
            return hidden_states[:, 0, :]

        if attention_mask is None:
            return jnp.mean(hidden_states, axis=1)
        pool_mask = attention_mask[..., None].astype(hidden_states.dtype)
        token_count = jnp.maximum(jnp.sum(pool_mask, axis=1), 1)
        return jnp.sum(hidden_states * pool_mask, axis=1) / token_count


def create_xlm_roberta(
    config: XLMRobertaConfig,
    *,
    seed: int = 0,
    mesh: Mesh | None = None,
    partition_axis: str | None = None,
) -> tuple[XLMRobertaEmbedding, Mesh | None]:
    """Create XLM-RoBERTa with optional one-axis FSDP placement."""
    if mesh is None and jax.device_count() > 1:
        mesh = create_fsdp_mesh()
    if mesh is not None and partition_axis is None:
        partition_axis = str(mesh.axis_names[0])
    if mesh is None:
        partition_axis = None

    with mesh_context(mesh):
        model = XLMRobertaEmbedding(
            config,
            rngs=nnx.Rngs(params=seed, dropout=seed + 1),
            partition_axis=partition_axis,
        )
    return model, mesh

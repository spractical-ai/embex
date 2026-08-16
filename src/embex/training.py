"""Reusable NNX training primitives for dual-encoder embedding models."""

from __future__ import annotations

from typing import Protocol

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from embex.utils.loss_functions import infonce_loss


class EmbeddingModel(Protocol):
    def __call__(
        self, input_ids: jax.Array, attention_mask: jax.Array | None = None
    ) -> jax.Array: ...


def create_optimizer(
    model: nnx.Module,
    learning_rate: float | optax.Schedule = 2e-5,
    *,
    max_grad_norm: float = 1.0,
) -> nnx.Optimizer:
    """Create the AdamW optimizer used by the original Qwen3 notebook."""
    transform = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adamw(learning_rate=learning_rate),
    )
    return nnx.Optimizer(model, transform, wrt=nnx.Param)


@nnx.jit
def contrastive_train_step(
    model: EmbeddingModel,
    optimizer: nnx.Optimizer,
    queries: jax.Array,
    keys: jax.Array,
    query_masks: jax.Array | None = None,
    key_masks: jax.Array | None = None,
    temperature: float = 0.05,
) -> jax.Array:
    """Run one tied-encoder InfoNCE update and return the scalar loss.

    NNX lifts ``model`` and ``optimizer`` through JIT, then applies the parameter and
    optimizer-state mutations to the passed objects after the compiled function returns.
    """

    def loss_fn(current_model: EmbeddingModel) -> jax.Array:
        query_embeddings = current_model(queries, query_masks).astype(jnp.float32)
        key_embeddings = current_model(keys, key_masks).astype(jnp.float32)
        return infonce_loss(query_embeddings, key_embeddings, temperature=temperature)

    loss, gradients = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, gradients)
    return loss

"""Loss functions for embedding-model training."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax


def infonce_loss(
    query_embeddings: jax.Array,
    key_embeddings: jax.Array,
    temperature: float = 0.05,
) -> jax.Array:
    """Compute one-directional in-batch-negative InfoNCE loss.

    Row ``i`` in ``query_embeddings`` is the positive pair for row ``i`` in
    ``key_embeddings``. Inputs must have equal batch dimensions.
    """
    if query_embeddings.shape[0] != key_embeddings.shape[0]:
        raise ValueError("Query and key embeddings must have the same batch size.")
    query_embeddings = query_embeddings / jnp.linalg.norm(
        query_embeddings, axis=-1, keepdims=True
    )
    key_embeddings = key_embeddings / jnp.linalg.norm(
        key_embeddings, axis=-1, keepdims=True
    )
    logits = jnp.matmul(query_embeddings, key_embeddings.T) / temperature
    labels = jnp.arange(query_embeddings.shape[0])
    return optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()

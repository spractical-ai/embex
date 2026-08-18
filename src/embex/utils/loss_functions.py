"""Loss functions for contrastive and masked-language-model training."""

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


def masked_language_model_loss(
    logits: jax.Array,
    labels: jax.Array,
    *,
    ignore_index: int = -100,
) -> jax.Array:
    """Average token cross-entropy over labels not equal to ``ignore_index``.

    MLM batches conventionally copy target token IDs into ``labels`` only at
    prediction positions and use ``-100`` everywhere else. A batch containing
    no prediction positions returns a finite zero loss.
    """
    if logits.ndim != labels.ndim + 1 or logits.shape[:-1] != labels.shape:
        raise ValueError(
            "MLM logits must have shape (*labels.shape, vocabulary_size)."
        )
    prediction_mask = labels != ignore_index
    safe_labels = jnp.where(prediction_mask, labels, 0)
    token_losses = optax.softmax_cross_entropy_with_integer_labels(
        logits.astype(jnp.float32), safe_labels
    )
    prediction_mask = prediction_mask.astype(jnp.float32)
    prediction_count = jnp.maximum(jnp.sum(prediction_mask), 1.0)
    return jnp.sum(token_losses * prediction_mask) / prediction_count

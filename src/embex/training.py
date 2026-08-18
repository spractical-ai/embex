"""Reusable NNX primitives for contrastive and masked-language training."""

from __future__ import annotations

from typing import Protocol

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from embex.utils.loss_functions import infonce_loss, masked_language_model_loss


class EmbeddingModel(Protocol):
    def __call__(
        self, input_ids: jax.Array, attention_mask: jax.Array | None = None
    ) -> jax.Array: ...


class XLMRobertaTrainingModel(Protocol):
    """The pooled and token-prediction interfaces exposed by XLM-RoBERTa."""

    def __call__(
        self,
        input_ids: jax.Array,
        attention_mask: jax.Array | None = None,
        *,
        deterministic: bool = True,
    ) -> jax.Array: ...

    def masked_language_model_logits(
        self,
        input_ids: jax.Array,
        attention_mask: jax.Array | None = None,
        *,
        deterministic: bool = True,
    ) -> jax.Array: ...


def create_optimizer(
    model: nnx.Module,
    learning_rate: float | optax.Schedule = 2e-5,
    *,
    max_grad_norm: float = 1.0,
) -> nnx.Optimizer:
    """Create the clipped AdamW optimizer used by EmbeX training steps."""
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


@nnx.jit
def masked_language_model_train_step(
    model: XLMRobertaTrainingModel,
    optimizer: nnx.Optimizer,
    input_ids: jax.Array,
    attention_mask: jax.Array,
    labels: jax.Array,
    ignore_index: int = -100,
) -> jax.Array:
    """Run one XLM-RoBERTa masked-language-model update."""

    def loss_fn(current_model: XLMRobertaTrainingModel) -> jax.Array:
        logits = current_model.masked_language_model_logits(
            input_ids, attention_mask, deterministic=False
        )
        return masked_language_model_loss(
            logits, labels, ignore_index=ignore_index
        )

    loss, gradients = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, gradients)
    return loss


@nnx.jit
def joint_contrastive_mlm_train_step(
    model: XLMRobertaTrainingModel,
    optimizer: nnx.Optimizer,
    queries: jax.Array,
    keys: jax.Array,
    query_masks: jax.Array,
    key_masks: jax.Array,
    mlm_input_ids: jax.Array,
    mlm_attention_mask: jax.Array,
    mlm_labels: jax.Array,
    temperature: float = 0.05,
    contrastive_weight: float = 1.0,
    mlm_weight: float = 1.0,
    ignore_index: int = -100,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Jointly update XLM-RoBERTa with InfoNCE and masked-token losses.

    Qwen3 does not implement the MLM-logits interface and remains on the
    contrastive-only ``contrastive_train_step`` path.
    """

    def loss_fn(
        current_model: XLMRobertaTrainingModel,
    ) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
        query_embeddings = current_model(
            queries, query_masks, deterministic=False
        ).astype(jnp.float32)
        key_embeddings = current_model(
            keys, key_masks, deterministic=False
        ).astype(jnp.float32)
        contrastive_loss = infonce_loss(
            query_embeddings, key_embeddings, temperature=temperature
        )
        logits = current_model.masked_language_model_logits(
            mlm_input_ids, mlm_attention_mask, deterministic=False
        )
        mlm_loss = masked_language_model_loss(
            logits, mlm_labels, ignore_index=ignore_index
        )
        total_loss = (
            contrastive_weight * contrastive_loss + mlm_weight * mlm_loss
        )
        return total_loss, (contrastive_loss, mlm_loss)

    (loss, component_losses), gradients = nnx.value_and_grad(
        loss_fn, has_aux=True
    )(model)
    optimizer.update(model, gradients)
    return loss, component_losses[0], component_losses[1]

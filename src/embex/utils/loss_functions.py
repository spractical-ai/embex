"""Loss functions for contrastive, distillation, and masked-LM training."""

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


def infonce_kd_loss(
    q_embeddings: jax.Array,
    k_embeddings: jax.Array,
    teacher_q_embeddings: jax.Array,
    teacher_k_embeddings: jax.Array,
    temperature: float = 0.05,
    kd_temperature: float = 0.07,
    kd_alpha: float = 0.25,
    false_negative_margin: float = 0.10,
    teacher_confirmation_margin: float = 0.05,
    duplicate_similarity_threshold: float = 0.999999,
    eps: float = 1e-6,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Compute false-negative-masked InfoNCE and relational KD losses.

    Row ``i`` in each query array is paired with row ``i`` in its corresponding
    key array. The hard loss masks student-suspected false negatives only when
    the teacher confirms them, and also masks duplicate teacher contexts. The
    relational KD term matches both query-to-key and key-to-query similarity
    distributions using the same ``kd_temperature`` for teacher and student.

    Teacher embedding width may differ from student embedding width, but the two
    arrays within each encoder must have matching widths and all four arrays must
    have the same batch size.
    """
    for name, embeddings in (
        ("q_embeddings", q_embeddings),
        ("k_embeddings", k_embeddings),
    ):
        if embeddings.ndim != 2:
            raise ValueError(f"{name} must have shape (batch, embedding).")
    if q_embeddings.shape != k_embeddings.shape:
        raise ValueError("Student query and key embeddings must have the same shape.")

    def l2_normalize(embeddings: jax.Array) -> jax.Array:
        embeddings = embeddings.astype(jnp.float32)
        norm = jnp.linalg.norm(embeddings, axis=-1, keepdims=True)
        return embeddings / jnp.maximum(norm, eps)

    q_emb = l2_normalize(q_embeddings)
    k_emb = l2_normalize(k_embeddings)
    student_similarity = jnp.matmul(q_emb, k_emb.T)
    labels = jnp.arange(student_similarity.shape[0])
    for name, embeddings in (
        ("teacher_q_embeddings", teacher_q_embeddings),
        ("teacher_k_embeddings", teacher_k_embeddings),
    ):
        if embeddings.ndim != 2:
            raise ValueError(f"{name} must have shape (batch, embedding).")
        if embeddings.shape[0] != q_embeddings.shape[0]:
            raise ValueError("Student and teacher embeddings must share a batch size.")
    if teacher_q_embeddings.shape[1] != teacher_k_embeddings.shape[1]:
        raise ValueError("Teacher query and key embedding widths must match.")

    teacher_q_emb = l2_normalize(teacher_q_embeddings)
    teacher_k_emb = l2_normalize(teacher_k_embeddings)
    teacher_similarity = jnp.matmul(teacher_q_emb, teacher_k_emb.T)

    batch_size = student_similarity.shape[0]
    diagonal = jnp.eye(batch_size, dtype=jnp.bool_)
    off_diagonal = ~diagonal

    detached_student_similarity = jax.lax.stop_gradient(student_similarity)
    detached_teacher_similarity = jax.lax.stop_gradient(teacher_similarity)
    student_positive_similarity = jnp.diag(detached_student_similarity)[:, None]
    teacher_positive_similarity = jnp.diag(detached_teacher_similarity)[:, None]

    student_suspects_false_negative = (
        detached_student_similarity
        > student_positive_similarity + false_negative_margin
    )
    teacher_confirms_similarity = (
        detached_teacher_similarity
        >= teacher_positive_similarity - teacher_confirmation_margin
    )
    teacher_confirmed_false_negative = (
        off_diagonal
        & student_suspects_false_negative
        & teacher_confirms_similarity
    )

    teacher_context_similarity = jax.lax.stop_gradient(
        jnp.matmul(teacher_k_emb, teacher_k_emb.T)
    )
    duplicate_context = off_diagonal & (
        teacher_context_similarity >= duplicate_similarity_threshold
    )
    false_negative_mask = teacher_confirmed_false_negative | duplicate_context
    valid_infonce_candidate = diagonal | (~false_negative_mask)

    hard_logits = student_similarity / temperature
    masked_hard_logits = jnp.where(
        valid_infonce_candidate,
        hard_logits,
        jnp.asarray(-1e9, dtype=hard_logits.dtype),
    )
    hard_loss = optax.softmax_cross_entropy_with_integer_labels(
        masked_hard_logits, labels
    ).mean()

    student_kd_logits = student_similarity / kd_temperature
    teacher_kd_logits = detached_teacher_similarity / kd_temperature

    def directional_kl(
        teacher_logits: jax.Array, student_logits: jax.Array
    ) -> jax.Array:
        teacher_log_probs = jax.nn.log_softmax(teacher_logits, axis=-1)
        teacher_probs = jnp.exp(teacher_log_probs)
        student_log_probs = jax.nn.log_softmax(student_logits, axis=-1)
        return jnp.sum(
            teacher_probs * (teacher_log_probs - student_log_probs), axis=-1
        ).mean()

    query_to_key_kd = directional_kl(teacher_kd_logits, student_kd_logits)
    key_to_query_kd = directional_kl(
        teacher_kd_logits.T, student_kd_logits.T
    )
    kd_loss = 0.5 * (query_to_key_kd + key_to_query_kd)
    total_loss = (1.0 - kd_alpha) * hard_loss + kd_alpha * kd_loss
    return total_loss, hard_loss, kd_loss


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

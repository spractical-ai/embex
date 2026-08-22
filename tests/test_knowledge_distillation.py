from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from embex.models.xlm_roberta import XLMRobertaConfig, create_xlm_roberta
from embex.training import create_optimizer, knowledge_distillation_train_step
from embex.utils.loss_functions import infonce_kd_loss, infonce_loss


def _tiny_model():
    config = XLMRobertaConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        max_position_embeddings=16,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        dtype=jnp.float32,
    )
    return create_xlm_roberta(config)[0]


def test_plain_infonce_remains_available_without_teacher_embeddings():
    queries = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    keys = queries

    loss = infonce_loss(queries, keys)

    assert loss.shape == ()
    assert bool(jnp.isfinite(loss))


def test_distillation_loss_masks_duplicate_contexts_and_combines_components():
    student_queries = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    student_keys = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    teacher_queries = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    teacher_keys = jnp.array([[1.0, 0.0], [1.0, 0.0]])

    total, hard, kd = infonce_kd_loss(
        student_queries,
        student_keys,
        teacher_queries,
        teacher_keys,
        kd_alpha=0.25,
    )

    np.testing.assert_allclose(hard, 0.0, atol=1e-7)
    assert kd > 0
    np.testing.assert_allclose(total, 0.75 * hard + 0.25 * kd, rtol=1e-6)


def test_teacher_embeddings_are_detached_from_the_distillation_objective():
    student_queries = jnp.array([[1.0, 0.0], [0.2, 0.8]])
    student_keys = jnp.array([[0.9, 0.1], [0.0, 1.0]])
    teacher_queries = jnp.array([[0.8, 0.2], [0.1, 0.9]])
    teacher_keys = jnp.array([[1.0, 0.0], [0.3, 0.7]])

    teacher_grad = jax.grad(
        lambda embeddings: infonce_kd_loss(
            student_queries,
            student_keys,
            embeddings,
            teacher_keys,
        )[0]
    )(teacher_queries)

    np.testing.assert_array_equal(teacher_grad, jnp.zeros_like(teacher_grad))


def test_knowledge_distillation_train_step_returns_finite_components():
    model = _tiny_model()
    optimizer = create_optimizer(model, learning_rate=1e-4)
    input_ids = jnp.array([[0, 4, 2], [0, 5, 2]])
    attention_mask = jnp.ones_like(input_ids)
    teacher_queries = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    teacher_keys = jnp.array([[0.9, 0.1, 0.0], [0.1, 0.9, 0.0]])

    total, hard, kd = knowledge_distillation_train_step(
        model,
        optimizer,
        input_ids,
        input_ids,
        attention_mask,
        attention_mask,
        teacher_queries,
        teacher_keys,
    )

    assert all(bool(jnp.isfinite(value)) for value in (total, hard, kd))
    np.testing.assert_allclose(total, 0.75 * hard + 0.25 * kd, rtol=1e-6)
    assert jax.tree_util.tree_leaves(optimizer.opt_state)

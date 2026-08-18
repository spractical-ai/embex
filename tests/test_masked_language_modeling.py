import jax
import jax.numpy as jnp
import numpy as np
import optax
from safetensors.numpy import save_file

from embex.models.xlm_roberta import (
    XLMRobertaConfig,
    create_xlm_roberta,
    load_xlm_roberta_weights,
    xlm_roberta_mlm_hf_state_dict,
)
from embex.training import (
    create_optimizer,
    joint_contrastive_mlm_train_step,
    masked_language_model_train_step,
)
from embex.utils.loss_functions import masked_language_model_loss


def tiny_config():
    return XLMRobertaConfig(
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


def test_mlm_loss_ignores_unselected_tokens_and_empty_batches():
    logits = jnp.array([[[2.0, 0.0], [100.0, -100.0]]])
    labels = jnp.array([[0, -100]])
    expected = optax.softmax_cross_entropy_with_integer_labels(
        logits[:, :1, :], labels[:, :1]
    ).mean()

    np.testing.assert_allclose(masked_language_model_loss(logits, labels), expected)
    assert masked_language_model_loss(
        logits, jnp.full_like(labels, -100)
    ).item() == 0.0


def test_mlm_logits_and_hugging_face_head_round_trip(tmp_path):
    source, _ = create_xlm_roberta(tiny_config(), seed=3)
    target, _ = create_xlm_roberta(tiny_config(), seed=9)
    weights = xlm_roberta_mlm_hf_state_dict(source)
    checkpoint_path = tmp_path / "model.safetensors"
    save_file(weights, str(checkpoint_path))
    load_xlm_roberta_weights(target, checkpoint_path)

    input_ids = jnp.array([[0, 4, 5, 2]])
    attention_mask = jnp.ones_like(input_ids)
    source_logits = source.masked_language_model_logits(input_ids, attention_mask)
    target_logits = target.masked_language_model_logits(input_ids, attention_mask)

    assert source_logits.shape == (1, 4, 32)
    np.testing.assert_allclose(source_logits, target_logits, rtol=1e-6, atol=1e-6)


def test_mlm_and_joint_training_steps_return_finite_losses():
    input_ids = jnp.array([[0, 4, 2], [0, 5, 2]])
    attention_mask = jnp.ones_like(input_ids)
    labels = jnp.array([[-100, 7, -100], [-100, 8, -100]])

    mlm_model, _ = create_xlm_roberta(tiny_config())
    mlm_optimizer = create_optimizer(mlm_model, 1e-4)
    mlm_loss = masked_language_model_train_step(
        mlm_model, mlm_optimizer, input_ids, attention_mask, labels
    )

    joint_model, _ = create_xlm_roberta(tiny_config())
    joint_optimizer = create_optimizer(joint_model, 1e-4)
    total, contrastive, joint_mlm = joint_contrastive_mlm_train_step(
        joint_model,
        joint_optimizer,
        input_ids,
        input_ids,
        attention_mask,
        attention_mask,
        input_ids,
        attention_mask,
        labels,
    )

    assert bool(jnp.isfinite(mlm_loss))
    assert all(bool(jnp.isfinite(value)) for value in (total, contrastive, joint_mlm))
    np.testing.assert_allclose(total, contrastive + joint_mlm, rtol=1e-6)
    assert jax.tree_util.tree_leaves(mlm_optimizer.opt_state)

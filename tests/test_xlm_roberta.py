import jax.numpy as jnp
import numpy as np

from embex.models.xlm_roberta import (
    XLMRobertaConfig,
    create_position_ids,
    create_xlm_roberta,
    update_xlm_roberta_weights,
    xlm_roberta_hf_state_dict,
)


def tiny_config(**overrides):
    values = {
        "vocab_size": 32,
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "max_position_embeddings": 16,
        "hidden_dropout_prob": 0.0,
        "attention_probs_dropout_prob": 0.0,
        "dtype": jnp.float32,
    }
    values.update(overrides)
    return XLMRobertaConfig(**values)


def test_position_ids_keep_padding_at_padding_index():
    input_ids = jnp.array([[0, 7, 2, 1, 1], [1, 0, 9, 2, 1]])
    positions = create_position_ids(input_ids, padding_idx=1)
    np.testing.assert_array_equal(
        positions, np.array([[2, 3, 4, 1, 1], [1, 2, 3, 4, 1]])
    )


def test_model_returns_masked_mean_and_contextual_states():
    model, _ = create_xlm_roberta(tiny_config())
    padded_ids = jnp.array([[0, 7, 2, 1, 1]])
    padded_mask = jnp.array([[1, 1, 1, 0, 0]])
    compact_ids = jnp.array([[0, 7, 2]])
    compact_mask = jnp.ones_like(compact_ids)

    assert model.encode_tokens(padded_ids, padded_mask).shape == (1, 5, 8)
    assert model(padded_ids, padded_mask).shape == (1, 8)
    np.testing.assert_allclose(
        model(padded_ids, padded_mask),
        model(compact_ids, compact_mask),
        rtol=1e-5,
        atol=1e-5,
    )


def test_training_dropout_path_accepts_explicit_nondeterministic_mode():
    model, _ = create_xlm_roberta(
        tiny_config(hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1)
    )
    input_ids = jnp.array([[0, 7, 2]])

    assert model(input_ids, jnp.ones_like(input_ids), deterministic=False).shape == (
        1,
        8,
    )


def test_hugging_face_weight_round_trip_accepts_roberta_prefix():
    source, _ = create_xlm_roberta(tiny_config(), seed=3)
    target, _ = create_xlm_roberta(tiny_config(), seed=9)
    weights = xlm_roberta_hf_state_dict(source)
    update_xlm_roberta_weights(
        {f"roberta.{name}": value for name, value in weights.items()}, target
    )

    input_ids = jnp.array([[0, 4, 5, 2]])
    attention_mask = jnp.ones_like(input_ids)
    np.testing.assert_allclose(
        source(input_ids, attention_mask),
        target(input_ids, attention_mask),
        rtol=1e-6,
        atol=1e-6,
    )


def test_presets_cover_base_and_large():
    base = XLMRobertaConfig.from_preset("FacebookAI/xlm-roberta-base")
    large = XLMRobertaConfig.from_preset("xlm_roberta_large")

    assert (base.hidden_size, base.num_hidden_layers) == (768, 12)
    assert (large.hidden_size, large.num_hidden_layers) == (1024, 24)

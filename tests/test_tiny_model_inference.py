"""End-to-end inference smoke tests for EmbeX's three public model calls.

Run with visible timing logs:

    uv run pytest -q -s tests/test_tiny_model_inference.py
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from embex.models.qwen3_embedding import (
    Qwen3EmbeddingConfig,
    create_qwen3_embedding,
)
from embex.models.xlm_roberta import XLMRobertaConfig, create_xlm_roberta


BATCH_SIZE = 2
SEQUENCE_LENGTH = 16
VOCAB_SIZE = 128
HIDDEN_SIZE = 32
BENCHMARK_RUNS = 10


@nnx.jit
def _embedding_call(model, input_ids, attention_mask):
    return model(input_ids, attention_mask)


@nnx.jit
def _mlm_call(model, input_ids, attention_mask):
    return model.masked_language_model_logits(input_ids, attention_mask)


def _qwen_inputs() -> tuple[jax.Array, jax.Array]:
    input_ids = jnp.array(
        [
            [11, 12, 13, 14, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [21, 22, 23, 24, 25, 26, 27, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=jnp.int32,
    )
    return input_ids, (input_ids != 0).astype(jnp.int32)


def _xlm_roberta_inputs() -> tuple[jax.Array, jax.Array]:
    input_ids = jnp.array(
        [
            [0, 11, 12, 13, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 21, 22, 23, 24, 25, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ],
        dtype=jnp.int32,
    )
    return input_ids, (input_ids != 1).astype(jnp.int32)


def _tiny_qwen():
    config = Qwen3EmbeddingConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        head_dim=8,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
    )
    return create_qwen3_embedding(config, seed=0)[0]


def _tiny_xlm_roberta():
    config = XLMRobertaConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=32,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
    )
    return create_xlm_roberta(config, seed=0)[0]


def _parameter_count(model: nnx.Module) -> int:
    return sum(
        leaf.size
        for leaf in jax.tree_util.tree_leaves(nnx.state(model, nnx.Param))
    )


def _benchmark(
    label: str,
    function: Callable[[nnx.Module, jax.Array, jax.Array], jax.Array],
    model: nnx.Module,
    input_ids: jax.Array,
    attention_mask: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    start = time.perf_counter()
    first_output = function(model, input_ids, attention_mask)
    jax.block_until_ready(first_output)
    first_call_ms = (time.perf_counter() - start) * 1_000

    samples = []
    repeated_output = first_output
    for _ in range(BENCHMARK_RUNS):
        start = time.perf_counter()
        repeated_output = function(model, input_ids, attention_mask)
        jax.block_until_ready(repeated_output)
        samples.append((time.perf_counter() - start) * 1_000)

    print(
        f"{label}: backend={jax.default_backend()} device={jax.devices()[0]} "
        f"params={_parameter_count(model):,} shape={first_output.shape} "
        f"dtype={first_output.dtype} first_call_ms={first_call_ms:.3f} "
        f"cached_mean_ms={statistics.mean(samples):.3f} "
        f"cached_median_ms={statistics.median(samples):.3f}"
    )
    return first_output, repeated_output


def _assert_valid_output(
    output: jax.Array,
    repeated_output: jax.Array,
    expected_shape: tuple[int, ...],
) -> None:
    assert output.shape == expected_shape
    assert output.dtype == jnp.float32
    assert bool(jnp.all(jnp.isfinite(output)))
    np.testing.assert_array_equal(output, repeated_output)


def test_tiny_qwen3_embedding_inference():
    model = _tiny_qwen()
    input_ids, attention_mask = _qwen_inputs()
    output, repeated_output = _benchmark(
        "qwen3_embedding",
        _embedding_call,
        model,
        input_ids,
        attention_mask,
    )
    _assert_valid_output(
        output, repeated_output, (BATCH_SIZE, HIDDEN_SIZE)
    )


def test_tiny_xlm_roberta_embedding_inference():
    model = _tiny_xlm_roberta()
    input_ids, attention_mask = _xlm_roberta_inputs()
    output, repeated_output = _benchmark(
        "xlm_roberta_embedding",
        _embedding_call,
        model,
        input_ids,
        attention_mask,
    )
    _assert_valid_output(
        output, repeated_output, (BATCH_SIZE, HIDDEN_SIZE)
    )


def test_tiny_xlm_roberta_mlm_inference():
    model = _tiny_xlm_roberta()
    input_ids, attention_mask = _xlm_roberta_inputs()
    logits, repeated_logits = _benchmark(
        "xlm_roberta_mlm",
        _mlm_call,
        model,
        input_ids,
        attention_mask,
    )
    _assert_valid_output(
        logits,
        repeated_logits,
        (BATCH_SIZE, SEQUENCE_LENGTH, VOCAB_SIZE),
    )

"""Eight-virtual-CPU tests for automatic EmbeX mesh and FSDP creation.

JAX reads its host-device count when its backend starts, so these checks run in
an isolated subprocess. This keeps the forced device count from affecting the
normal test suite.

Run with visible mesh and timing logs:

    uv run pytest -q -s -p no:cacheprovider tests/test_tiny_model_inference_8_devices.py
"""

from __future__ import annotations

import os
import re
import subprocess
import sys


_EIGHT_DEVICE_SCRIPT = r"""
import statistics
import time

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from embex.models.qwen3_embedding import (
    Qwen3EmbeddingConfig,
    create_qwen3_embedding,
)
from embex.models.xlm_roberta import XLMRobertaConfig, create_xlm_roberta
from embex.utils.distributed import shard_batch


BATCH_SIZE = 8
SEQUENCE_LENGTH = 16
VOCAB_SIZE = 128
HIDDEN_SIZE = 32
BENCHMARK_RUNS = 5


@nnx.jit
def embedding_call(model, input_ids, attention_mask):
    return model(input_ids, attention_mask)


@nnx.jit
def mlm_call(model, input_ids, attention_mask):
    return model.masked_language_model_logits(input_ids, attention_mask)


def qwen_inputs():
    input_ids = jnp.zeros((BATCH_SIZE, SEQUENCE_LENGTH), dtype=jnp.int32)
    tokens = jnp.arange(BATCH_SIZE * 6, dtype=jnp.int32).reshape(BATCH_SIZE, 6)
    input_ids = input_ids.at[:, :6].set((tokens % (VOCAB_SIZE - 1)) + 1)
    attention_mask = jnp.zeros_like(input_ids).at[:, :6].set(1)
    return input_ids, attention_mask


def xlm_roberta_inputs():
    input_ids = jnp.ones((BATCH_SIZE, SEQUENCE_LENGTH), dtype=jnp.int32)
    tokens = jnp.arange(BATCH_SIZE * 5, dtype=jnp.int32).reshape(BATCH_SIZE, 5)
    input_ids = input_ids.at[:, 0].set(0)
    input_ids = input_ids.at[:, 1:6].set((tokens % (VOCAB_SIZE - 3)) + 3)
    input_ids = input_ids.at[:, 6].set(2)
    attention_mask = jnp.zeros_like(input_ids).at[:, :7].set(1)
    return input_ids, attention_mask


def assert_automatic_mesh(mesh):
    assert mesh is not None
    assert mesh.devices.size == 8
    assert tuple(str(name) for name in mesh.axis_names) == ("fsdp",)


def parameter_sharding_summary(model):
    parameters = jax.tree_util.tree_leaves(nnx.state(model, nnx.Param))
    specs = {
        str(getattr(getattr(parameter, "sharding", None), "spec", None))
        for parameter in parameters
    }
    partitioned = {spec for spec in specs if "fsdp" in spec}
    assert partitioned, f"No parameters use the automatic FSDP axis: {specs}"
    return len(parameters), sorted(specs)


def benchmark(label, function, model, mesh, input_ids, attention_mask, shape):
    sharded_ids = shard_batch(input_ids, mesh)
    sharded_mask = shard_batch(attention_mask, mesh)
    assert len(sharded_ids.addressable_shards) == 8
    assert len(sharded_mask.addressable_shards) == 8

    start = time.perf_counter()
    output = function(model, sharded_ids, sharded_mask)
    jax.block_until_ready(output)
    first_call_ms = (time.perf_counter() - start) * 1_000

    samples = []
    repeated_output = output
    for _ in range(BENCHMARK_RUNS):
        start = time.perf_counter()
        repeated_output = function(model, sharded_ids, sharded_mask)
        jax.block_until_ready(repeated_output)
        samples.append((time.perf_counter() - start) * 1_000)

    assert output.shape == shape
    assert output.dtype == jnp.float32
    assert bool(jnp.all(jnp.isfinite(output)))
    np.testing.assert_array_equal(output, repeated_output)
    assert len(output.addressable_shards) == 8

    parameter_count, sharding_specs = parameter_sharding_summary(model)
    print(
        f"{label}: devices={jax.device_count()} mesh={mesh.shape} "
        f"axis_names={mesh.axis_names} parameter_leaves={parameter_count} "
        f"parameter_specs={sharding_specs} input_spec={sharded_ids.sharding.spec} "
        f"output_spec={output.sharding.spec} shape={output.shape} "
        f"first_call_ms={first_call_ms:.3f} "
        f"cached_mean_ms={statistics.mean(samples):.3f} "
        f"cached_median_ms={statistics.median(samples):.3f}"
    )


assert jax.default_backend() == "cpu"
assert jax.device_count() == 8, jax.devices()
print(f"backend={jax.default_backend()} devices={jax.devices()}")

qwen_config = Qwen3EmbeddingConfig(
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
qwen, qwen_mesh = create_qwen3_embedding(qwen_config, seed=0)
assert_automatic_mesh(qwen_mesh)
qwen_ids, qwen_mask = qwen_inputs()
benchmark(
    "qwen3_embedding_8cpu",
    embedding_call,
    qwen,
    qwen_mesh,
    qwen_ids,
    qwen_mask,
    (BATCH_SIZE, HIDDEN_SIZE),
)

xlmr_config = XLMRobertaConfig(
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
xlmr, xlmr_mesh = create_xlm_roberta(xlmr_config, seed=0)
assert_automatic_mesh(xlmr_mesh)
xlmr_ids, xlmr_mask = xlm_roberta_inputs()
benchmark(
    "xlm_roberta_embedding_8cpu",
    embedding_call,
    xlmr,
    xlmr_mesh,
    xlmr_ids,
    xlmr_mask,
    (BATCH_SIZE, HIDDEN_SIZE),
)
benchmark(
    "xlm_roberta_mlm_8cpu",
    mlm_call,
    xlmr,
    xlmr_mesh,
    xlmr_ids,
    xlmr_mask,
    (BATCH_SIZE, SEQUENCE_LENGTH, VOCAB_SIZE),
)
"""


def test_tiny_models_create_and_run_on_automatic_eight_cpu_mesh():
    environment = os.environ.copy()
    xla_flags = re.sub(
        r"(?:^|\s)--xla_force_host_platform_device_count=\d+",
        "",
        environment.get("XLA_FLAGS", ""),
    ).strip()
    environment["XLA_FLAGS"] = (
        f"{xla_flags} --xla_force_host_platform_device_count=8".strip()
    )
    environment["JAX_PLATFORMS"] = "cpu"
    environment["PYTHONPATH"] = os.pathsep.join(
        path for path in sys.path if path
    )

    result = subprocess.run(
        [sys.executable, "-c", _EIGHT_DEVICE_SCRIPT],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=180,
    )
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    assert result.returncode == 0

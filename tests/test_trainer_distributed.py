"""Exercise trainer updates and validation on an isolated multi-device CPU mesh."""

import os
import re
import subprocess
import sys


def test_all_trainer_objectives_on_two_cpu_devices():
    script = r'''
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import AxisType

from embex import (
    ContrastiveBatch, EmbeddingTrainer, JointContrastiveMLMBatch,
    KnowledgeDistillationBatch, MLMBatch,
)
from embex.models.qwen3_embedding import Qwen3EmbeddingConfig, create_qwen3_embedding
from embex.models.xlm_roberta import XLMRobertaConfig, create_xlm_roberta
from embex.training import create_optimizer

assert jax.device_count() == 2
common = dict(
    vocab_size=32, hidden_size=8, intermediate_size=16, num_hidden_layers=1,
    num_attention_heads=2, dtype=jnp.float32,
)
qwen, qwen_mesh = create_qwen3_embedding(
    Qwen3EmbeddingConfig(**common, head_dim=4, num_key_value_heads=1),
)
xlmr, xlmr_mesh = create_xlm_roberta(
    XLMRobertaConfig(**common, max_position_embeddings=16),
)
ids = np.array([[0, 4, 2], [0, 5, 2]], dtype=np.int32)
mask = np.ones_like(ids)
pairs = ContrastiveBatch(ids, ids, mask, mask)
mlm = MLMBatch(ids, mask, np.array([[-100, 6, -100], [-100, 7, -100]], dtype=np.int32))
teacher = np.eye(2, 3, dtype=np.float32)
kd = KnowledgeDistillationBatch(pairs, teacher, teacher)
joint = JointContrastiveMLMBatch(pairs, mlm)
for model, mesh in [(qwen, qwen_mesh), (xlmr, xlmr_mesh)]:
    assert mesh is not None and mesh.size == 2
    cases = [("contrastive", pairs), ("knowledge_distillation", kd)]
    if model is xlmr:
        cases += [("masked_language_modeling", mlm), ("joint_contrastive_mlm", joint)]
    for objective, batch in cases:
        optimizer = create_optimizer(model)
        mesh_kwargs = {} if objective == "contrastive" else {"mesh": mesh}
        trainer = EmbeddingTrainer(
            model, optimizer, objective=objective, train_dataset=[batch], **mesh_kwargs,
        )
        assert trainer.mesh == mesh
        metrics = trainer.train()[0]
        assert all(np.isfinite(value) for value in metrics.values()), metrics
        assert int(optimizer.step.get_value()) == 1
        parameters = jax.tree.leaves(nnx.state(model, nnx.Param))
        assert any(
            "fsdp" in str(getattr(value.sharding, "spec", None)) for value in parameters
        )

# Each input's leading dimension must be divisible by the mesh size.
bad = MLMBatch(ids[:1], mask[:1], mlm.labels[:1])
for objective, batch in [
    ("masked_language_modeling", bad),
    ("joint_contrastive_mlm", JointContrastiveMLMBatch(pairs, bad)),
]:
    optimizer = create_optimizer(xlmr)
    trainer = EmbeddingTrainer(
        xlmr, optimizer, mesh=xlmr_mesh, objective=objective, train_dataset=[batch],
    )
    try:
        trainer.train()
    except ValueError as error:
        assert "must be divisible" in str(error), str(error)
    else:
        raise AssertionError("Invalid sharded batch was accepted")
    assert int(optimizer.step.get_value()) == 0

two_axes = jax.make_mesh((1, 2), ("data", "model"), axis_types=(AxisType.Auto, AxisType.Auto))
try:
    EmbeddingTrainer(xlmr, create_optimizer(xlmr), mesh=two_axes)
except ValueError as error:
    assert "one-axis mesh" in str(error)
else:
    raise AssertionError("Unsupported mesh was accepted")

other_mesh = jax.make_mesh((2,), ("other",), axis_types=(AxisType.Auto,))
try:
    EmbeddingTrainer(xlmr, create_optimizer(xlmr), mesh=other_mesh)
except ValueError as error:
    assert "must match the model parameter mesh" in str(error)
else:
    raise AssertionError("Mismatched mesh was accepted")
'''
    environment = os.environ.copy()
    flags = re.sub(
        r"(?:^|\s)--xla_force_host_platform_device_count=\d+", "",
        environment.get("XLA_FLAGS", ""),
    ).strip()
    environment["XLA_FLAGS"] = f"{flags} --xla_force_host_platform_device_count=2".strip()
    environment["JAX_PLATFORMS"] = "cpu"
    environment["PYTHONPATH"] = os.pathsep.join(path for path in sys.path if path)
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True,
        env=environment, timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr

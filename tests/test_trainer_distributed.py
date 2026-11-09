"""Exercise trainer updates and validation on an isolated multi-device CPU mesh."""

import pytest


@pytest.mark.parametrize("parallelism", ["fsdp", "dp"])
def test_all_trainer_objectives_on_two_cpu_devices(parallelism, run_on_cpu_devices):
    script = r'''
import sys
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
from embex.models.qwen3_embedding import qwen3_embedding_hf_state_dict
from embex.models.qwen3_embedding.loading_weights import update_qwen3_embedding_weights
from embex.models.xlm_roberta import (
    XLMRobertaConfig, create_xlm_roberta,
    xlm_roberta_mlm_hf_state_dict, update_xlm_roberta_weights,
)
from embex.training import create_optimizer
from embex.utils.distributed import get_model_mesh

assert jax.device_count() == 2
parallelism = sys.argv[1]
common = dict(
    vocab_size=32, hidden_size=8, intermediate_size=16, num_hidden_layers=1,
    num_attention_heads=2, dtype=jnp.float32,
)
qwen, qwen_mesh = create_qwen3_embedding(
    Qwen3EmbeddingConfig(**common, head_dim=4, num_key_value_heads=1),
    parallelism=parallelism,
)
xlmr, xlmr_mesh = create_xlm_roberta(
    XLMRobertaConfig(**common, max_position_embeddings=16,
                     hidden_dropout_prob=0, attention_probs_dropout_prob=0),
    parallelism=parallelism,
)
# A full-batch reference on one device detects per-device losses/gradients and
# lost in-batch negatives. Dropout is disabled to make the comparison numerical.
reference_mesh = jax.make_mesh(
    (1,), ("reference",), devices=jax.devices()[:1], axis_types=(AxisType.Auto,),
)
qwen_reference, _ = create_qwen3_embedding(
    qwen.config, mesh=reference_mesh, parallelism="dp",
)
xlmr_reference, _ = create_xlm_roberta(
    xlmr.config, mesh=reference_mesh, parallelism="dp",
)
# Exercise training after the same adapters used to load pretrained checkpoints.
update_qwen3_embedding_weights(qwen3_embedding_hf_state_dict(qwen), qwen)
update_xlm_roberta_weights(xlm_roberta_mlm_hf_state_dict(xlmr), xlmr)

def assert_placement(model, optimizer=None):
    parameters = jax.tree.leaves(nnx.state(model, nnx.Param))
    assert all(len(value.addressable_shards) == 2 for value in parameters)
    if parallelism == "dp":
        assert all(value.sharding.is_fully_replicated for value in parameters)
        if optimizer is not None:
            moments = [value for value in jax.tree.leaves(nnx.state(optimizer))
                       if getattr(value, "ndim", 0) > 0]
            assert moments
            assert all(value.sharding.is_fully_replicated for value in moments)
            assert all(len(value.addressable_shards) == 2 for value in moments)
        for value in parameters:
            for shard in value.addressable_shards:
                np.testing.assert_array_equal(shard.data, value)
    else:
        assert any(not value.sharding.is_fully_replicated for value in parameters)
        if optimizer is not None:
            assert any(not value.sharding.is_fully_replicated
                       for value in jax.tree.leaves(nnx.state(optimizer)))

ids = np.array([[0, 4, 2], [0, 5, 2]], dtype=np.int32)
mask = np.ones_like(ids)
pairs = ContrastiveBatch(ids, ids, mask, mask)
mlm = MLMBatch(ids, mask, np.array([[-100, 6, -100], [-100, 7, -100]], dtype=np.int32))
teacher = np.eye(2, 3, dtype=np.float32)
kd = KnowledgeDistillationBatch(pairs, teacher, teacher)
joint = JointContrastiveMLMBatch(pairs, mlm)
for model, mesh, reference in [
    (qwen, qwen_mesh, qwen_reference), (xlmr, xlmr_mesh, xlmr_reference),
]:
    assert mesh is not None and mesh.size == 2
    assert tuple(mesh.axis_names) == (parallelism,)
    assert get_model_mesh(model) == mesh
    assert_placement(model)
    cases = [("contrastive", pairs), ("knowledge_distillation", kd)]
    if model is xlmr:
        cases += [("masked_language_modeling", mlm), ("joint_contrastive_mlm", joint)]
    for objective, batch in cases:
        optimizer = create_optimizer(model, 1e-3)
        reference_optimizer = create_optimizer(reference, 1e-3)
        assert_placement(model, optimizer)
        mesh_kwargs = {} if objective == "contrastive" else {"mesh": mesh}
        trainer = EmbeddingTrainer(
            model, optimizer, objective=objective, train_dataset=[batch], **mesh_kwargs,
        )
        assert trainer.mesh == mesh
        reference_trainer = EmbeddingTrainer(
            reference, reference_optimizer, objective=objective,
        )
        for step in range(2):
            metrics = trainer.train()[0]
            expected = reference_trainer.train([batch])[0]
            assert all(np.isfinite(value) for value in metrics.values()), metrics
            for name in metrics:
                np.testing.assert_allclose(metrics[name], expected[name], rtol=2e-5, atol=2e-6)
            assert int(optimizer.step.get_value()) == step + 1
            assert_placement(model, optimizer)
            for actual, expected in zip(
                jax.tree.leaves(nnx.state(model, nnx.Param)),
                jax.tree.leaves(nnx.state(reference, nnx.Param)), strict=True,
            ):
                np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-6)

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
    run_on_cpu_devices(script, 2, parallelism)

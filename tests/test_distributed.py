"""Placement options and backwards compatibility for both model factories."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from embex.models.qwen3_embedding import Qwen3EmbeddingConfig, create_qwen3_embedding
from embex.models.xlm_roberta import XLMRobertaConfig, create_xlm_roberta
from embex.utils.distributed import create_fsdp_mesh, create_mesh, shard_batch


@pytest.fixture(params=["qwen3", "xlm_roberta"])
def factory_and_config(request):
    common = dict(
        vocab_size=32, hidden_size=8, intermediate_size=16,
        num_hidden_layers=1, num_attention_heads=2, dtype=jnp.float32,
    )
    if request.param == "qwen3":
        return create_qwen3_embedding, Qwen3EmbeddingConfig(
            **common, head_dim=4, num_key_value_heads=1,
        )
    return create_xlm_roberta, XLMRobertaConfig(
        **common, max_position_embeddings=16,
    )


@pytest.mark.parametrize("parallelism", ["fsdp", "dp"])
def test_single_device_modes_preserve_inference(factory_and_config, parallelism):
    if jax.device_count() != 1:
        pytest.skip("Checks the single-device fallback.")
    factory, config = factory_and_config
    model, mesh = factory(config, parallelism=parallelism)
    reference, default_mesh = factory(config)
    assert mesh is None and default_mesh is None
    ids = np.array([[0, 4, 2], [0, 5, 2]], dtype=np.int32)
    mask = np.ones_like(ids)
    np.testing.assert_array_equal(
        model(shard_batch(ids, mesh), shard_batch(mask, mesh)), reference(ids, mask),
    )


@pytest.mark.parametrize("parallelism", ["ddp", "FSDP", "", None])
def test_invalid_parallelism_is_rejected(factory_and_config, parallelism):
    factory, config = factory_and_config
    with pytest.raises(ValueError, match="parallelism must be 'fsdp' or 'dp'"):
        factory(config, parallelism=parallelism)


def test_mesh_helper_validates_mode_and_device_count():
    with pytest.raises(ValueError, match="parallelism"):
        create_mesh(parallelism="other")
    for create in (create_mesh, create_fsdp_mesh):
        with pytest.raises(ValueError, match="at least two"):
            create(devices=jax.devices()[:1])


@pytest.mark.parametrize("parallelism", ["fsdp", "dp"])
def test_explicit_mesh_uses_mode_with_custom_axis(parallelism, run_on_cpu_devices):
    run_on_cpu_devices(r'''
import sys
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from embex.models.qwen3_embedding import Qwen3EmbeddingConfig, create_qwen3_embedding
from embex.models.qwen3_embedding.loading_weights import update_qwen3_embedding_weights
from embex.models.qwen3_embedding import qwen3_embedding_hf_state_dict
from embex.models.xlm_roberta import (
    XLMRobertaConfig, create_xlm_roberta,
    update_xlm_roberta_weights, xlm_roberta_mlm_hf_state_dict,
)
from embex.utils.distributed import create_fsdp_mesh, create_mesh, get_model_mesh, shard_batch

parallelism = sys.argv[1]
mesh = create_mesh(parallelism=parallelism, axis_name="custom")
assert mesh.size == 2 and tuple(mesh.axis_names) == ("custom",)
assert tuple(create_fsdp_mesh().axis_names) == ("fsdp",)
assert tuple(create_fsdp_mesh(axis_name="legacy").axis_names) == ("legacy",)
common = dict(vocab_size=32, hidden_size=8, intermediate_size=16,
              num_hidden_layers=1, num_attention_heads=2, dtype=jnp.float32)
cases = [
    (create_qwen3_embedding,
     Qwen3EmbeddingConfig(**common, head_dim=4, num_key_value_heads=1),
     qwen3_embedding_hf_state_dict, update_qwen3_embedding_weights),
    (create_xlm_roberta, XLMRobertaConfig(**common, max_position_embeddings=16),
     xlm_roberta_mlm_hf_state_dict, update_xlm_roberta_weights),
]
for factory, config, export, load in cases:
    model, actual_mesh = factory(config, mesh=mesh, parallelism=parallelism,
                                 partition_axis="custom")
    source, _ = factory(config, mesh=mesh, parallelism=parallelism, seed=7)
    assert actual_mesh == mesh and get_model_mesh(model) == mesh
    load(export(source), model)
    parameters = jax.tree.leaves(nnx.state(model, nnx.Param))
    assert all(value.sharding.mesh == mesh for value in parameters)
    if parallelism == "dp":
        assert all(value.sharding.is_fully_replicated for value in parameters)
    else:
        assert any(not value.sharding.is_fully_replicated for value in parameters)
    ids = shard_batch(np.array([[0, 4, 2], [0, 5, 2]], dtype=np.int32), mesh)
    masks = shard_batch(np.ones((2, 3), dtype=np.int32), mesh)
    assert tuple(ids.sharding.spec) == ("custom",)
    np.testing.assert_allclose(model(ids, masks), source(ids, masks), atol=1e-6)
    for bad_mesh, axis, error, message in [
        (mesh, "wrong", ValueError, "must match"),
        (jax.make_mesh((1, 2), ("a", "b"), axis_types=(jax.sharding.AxisType.Auto,) * 2),
         None, ValueError, "one-axis"),
        ("invalid", None, TypeError, "JAX Mesh"),
    ]:
        try:
            factory(config, mesh=bad_mesh, partition_axis=axis, parallelism=parallelism)
        except error as exc:
            assert message in str(exc)
        else:
            raise AssertionError("Invalid placement was accepted")
''', 2, parallelism)

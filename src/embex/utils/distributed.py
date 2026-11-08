"""Device-agnostic mesh and sharding helpers."""

from __future__ import annotations

from contextlib import nullcontext
from typing import ContextManager, Sequence

import jax
from flax import nnx
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec


def create_fsdp_mesh(
    devices: Sequence[jax.Device] | None = None, *, axis_name: str = "fsdp"
) -> Mesh:
    """Create a one-axis mesh over the supplied accelerator devices.

    The function does not assume CPU, GPU, or TPU.  It is only called automatically
    for multi-device JAX runtimes; on one device model creation is left unpartitioned.
    """
    devices = tuple(jax.devices() if devices is None else devices)
    if len(devices) < 2:
        raise ValueError("FSDP requires at least two JAX devices.")
    return jax.make_mesh(
        (len(devices),), (axis_name,), devices=devices, axis_types=(AxisType.Auto,)
    )


def mesh_context(mesh: Mesh | None) -> ContextManager[None]:
    """Return a no-op context for one device or activate a supplied mesh."""
    return nullcontext() if mesh is None else jax.set_mesh(mesh)


def get_model_mesh(model: nnx.Module) -> Mesh | None:
    """Infer the common mesh of model parameters without transferring arrays."""
    meshes = set()
    for parameter in jax.tree.leaves(nnx.state(model, nnx.Param)):
        sharding = getattr(parameter, "sharding", None)
        if isinstance(sharding, NamedSharding):
            meshes.add(sharding.mesh)
    if len(meshes) > 1:
        raise ValueError("Model parameters must use the same mesh.")
    return next(iter(meshes), None)


def shard_batch(
    batch: jax.Array, mesh: Mesh | None, *, axis_name: str | None = None
) -> jax.Array:
    """Shard the leading batch dimension when a compatible one-axis mesh is used."""
    if mesh is None:
        return jax.device_put(batch)
    axis_name = str(mesh.axis_names[0]) if axis_name is None else axis_name
    return jax.device_put(batch, NamedSharding(mesh, PartitionSpec(axis_name)))

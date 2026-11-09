"""Device-agnostic mesh and sharding helpers."""

from __future__ import annotations

from contextlib import nullcontext
from typing import ContextManager, Literal, Sequence

import jax
from flax import nnx
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec


Parallelism = Literal["fsdp", "dp"]


def _validate_parallelism(parallelism: str) -> None:
    if parallelism not in ("fsdp", "dp"):
        raise ValueError("parallelism must be 'fsdp' or 'dp'.")


def create_mesh(
    devices: Sequence[jax.Device] | None = None,
    *,
    parallelism: Parallelism = "fsdp",
    axis_name: str | None = None,
) -> Mesh:
    """Create a one-axis FSDP or data-parallel mesh over the supplied devices."""
    _validate_parallelism(parallelism)
    devices = tuple(jax.devices() if devices is None else devices)
    if len(devices) < 2:
        raise ValueError("Parallel execution requires at least two JAX devices.")
    return jax.make_mesh(
        (len(devices),), (axis_name or parallelism,), devices=devices,
        axis_types=(AxisType.Auto,),
    )


def create_fsdp_mesh(
    devices: Sequence[jax.Device] | None = None, *, axis_name: str = "fsdp"
) -> Mesh:
    """Create a one-axis mesh over the supplied accelerator devices.

    The function does not assume CPU, GPU, or TPU.  It is only called automatically
    for multi-device JAX runtimes; on one device model creation is left unpartitioned.
    """
    return create_mesh(devices, parallelism="fsdp", axis_name=axis_name)


def _resolve_model_mesh(
    mesh: Mesh | None, partition_axis: str | None, parallelism: Parallelism
) -> tuple[Mesh | None, str | None]:
    """Resolve the common factory placement contract for both model families."""
    _validate_parallelism(parallelism)
    if mesh is None and jax.device_count() > 1:
        mesh = create_mesh(parallelism=parallelism)
    if mesh is None:
        return None, None
    if not isinstance(mesh, Mesh):
        raise TypeError("mesh must be a JAX Mesh or None.")
    if len(mesh.axis_names) != 1:
        raise ValueError("Model creation requires a one-axis mesh.")
    axis = str(mesh.axis_names[0])
    if partition_axis is not None and partition_axis != axis:
        raise ValueError("partition_axis must match the mesh axis.")
    return mesh, axis


def _replicate_parameters(model: nnx.Module, mesh: Mesh) -> None:
    """Replicate every parameter and retain this layout in NNX optimizer metadata."""
    sharding = NamedSharding(mesh, PartitionSpec())
    state = nnx.state(model, nnx.Param)
    for variable in jax.tree.leaves(
        state, is_leaf=lambda value: isinstance(value, nnx.Variable)
    ):
        variable.set_metadata(out_sharding=(None,) * variable.ndim)
        variable.set_value(jax.device_put(variable.get_value(), sharding))
    nnx.update(model, state)


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

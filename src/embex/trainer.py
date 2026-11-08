"""A shared training loop for embedding and masked-language objectives."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from inspect import signature
from math import isfinite
from numbers import Integral, Real
from typing import Any, Callable

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh

from embex.training import (
    contrastive_train_step,
    joint_contrastive_mlm_train_step,
    knowledge_distillation_train_step,
    masked_language_model_train_step,
)
from embex.utils.distributed import get_model_mesh, mesh_context, shard_batch


def _matrix(name: str, value: Any, *, integer: bool = False) -> None:
    shape = getattr(value, "shape", ())
    if len(shape) != 2 or any(size < 1 for size in shape):
        raise ValueError(f"{name} must have a nonempty two-dimensional shape.")
    dtype = getattr(value, "dtype", None)
    if dtype is None:
        raise TypeError(f"{name} must be a NumPy or JAX array.")
    if integer and not jnp.issubdtype(dtype, jnp.integer):
        raise TypeError(f"{name} must contain integer token IDs or labels.")


def _token_pair(name: str, input_ids: Any, attention_mask: Any) -> None:
    _matrix(f"{name}_input_ids", input_ids, integer=True)
    _matrix(f"{name}_attention_mask", attention_mask)
    if input_ids.shape != attention_mask.shape:
        raise ValueError(
            f"{name}_input_ids and {name}_attention_mask must have the same shape."
        )
    if not (
        jnp.issubdtype(attention_mask.dtype, jnp.integer)
        or jnp.issubdtype(attention_mask.dtype, jnp.floating)
        or jnp.issubdtype(attention_mask.dtype, jnp.bool_)
    ):
        raise TypeError(f"{name}_attention_mask must be numeric or boolean.")


@dataclass(frozen=True)
class ContrastiveBatch:
    """Aligned query/positive-document token arrays with explicit padding masks."""

    query_input_ids: Any
    key_input_ids: Any
    query_attention_mask: Any
    key_attention_mask: Any

    def __post_init__(self) -> None:
        _token_pair("query", self.query_input_ids, self.query_attention_mask)
        _token_pair("key", self.key_input_ids, self.key_attention_mask)
        if self.query_input_ids.shape[0] != self.key_input_ids.shape[0]:
            raise ValueError("Query and key batches must contain the same number of rows.")

    @classmethod
    def from_tokenizer(
        cls,
        tokenizer: Any,
        queries: Sequence[str],
        keys: Sequence[str],
        *,
        max_length: int = 512,
    ) -> ContrastiveBatch:
        """Tokenize aligned text pairs outside JAX using a HF tokenizer."""
        if len(queries) != len(keys):
            raise ValueError("queries and keys must have equal lengths.")
        if len(queries) == 0:
            raise ValueError("queries and keys must not be empty.")
        _positive_integer("max_length", max_length)
        tokenizer_kwargs = {
            "padding": "max_length",
            "truncation": True,
            "max_length": max_length,
            "return_tensors": "np",
        }
        query_tokens = tokenizer(queries, **tokenizer_kwargs)
        key_tokens = tokenizer(keys, **tokenizer_kwargs)
        return cls(
            query_input_ids=query_tokens["input_ids"],
            key_input_ids=key_tokens["input_ids"],
            query_attention_mask=query_tokens["attention_mask"],
            key_attention_mask=key_tokens["attention_mask"],
        )

    def _step_inputs(self) -> dict[str, Any]:
        return {
            "queries": self.query_input_ids,
            "keys": self.key_input_ids,
            "query_masks": self.query_attention_mask,
            "key_masks": self.key_attention_mask,
        }


@dataclass(frozen=True)
class KnowledgeDistillationBatch:
    """Paired tokens and precomputed, row-aligned teacher embeddings.

    Teacher width may differ from student width. The teacher's query and key
    widths must match each other.
    """

    contrastive: ContrastiveBatch
    teacher_query_embeddings: Any
    teacher_key_embeddings: Any

    def __post_init__(self) -> None:
        if not isinstance(self.contrastive, ContrastiveBatch):
            raise TypeError("contrastive must be a ContrastiveBatch.")
        for name, embeddings in (
            ("teacher_query_embeddings", self.teacher_query_embeddings),
            ("teacher_key_embeddings", self.teacher_key_embeddings),
        ):
            _matrix(name, embeddings)
            if not jnp.issubdtype(embeddings.dtype, jnp.floating):
                raise TypeError(f"{name} must contain floating-point embeddings.")
            if embeddings.shape[0] != self.contrastive.query_input_ids.shape[0]:
                raise ValueError("Teacher and student batches must have the same row count.")
        if self.teacher_query_embeddings.shape != self.teacher_key_embeddings.shape:
            raise ValueError("Teacher query and key embeddings must have the same shape.")

    def _step_inputs(self) -> dict[str, Any]:
        return {
            **self.contrastive._step_inputs(),
            "teacher_q_embeddings": self.teacher_query_embeddings,
            "teacher_k_embeddings": self.teacher_key_embeddings,
        }


@dataclass(frozen=True)
class MLMBatch:
    """Masked input tokens and target labels; unselected labels default to -100.

    Masking is performed by the caller's data pipeline, outside the trainer.
    """

    input_ids: Any
    attention_mask: Any
    labels: Any

    def __post_init__(self) -> None:
        _token_pair("mlm", self.input_ids, self.attention_mask)
        _matrix("labels", self.labels, integer=True)
        if self.labels.shape != self.input_ids.shape:
            raise ValueError("MLM labels and input_ids must have the same shape.")

    def _step_inputs(self) -> dict[str, Any]:
        return {
            "input_ids": self.input_ids,
            "attention_mask": self.attention_mask,
            "labels": self.labels,
        }


@dataclass(frozen=True)
class JointContrastiveMLMBatch:
    """Contrastive and MLM inputs for one joint update.

    The two objectives may use different batch sizes and sequence lengths.
    """

    contrastive: ContrastiveBatch
    mlm: MLMBatch

    def __post_init__(self) -> None:
        if not isinstance(self.contrastive, ContrastiveBatch):
            raise TypeError("contrastive must be a ContrastiveBatch.")
        if not isinstance(self.mlm, MLMBatch):
            raise TypeError("mlm must be an MLMBatch.")

    def _step_inputs(self) -> dict[str, Any]:
        return {
            **self.contrastive._step_inputs(),
            "mlm_input_ids": self.mlm.input_ids,
            "mlm_attention_mask": self.mlm.attention_mask,
            "mlm_labels": self.mlm.labels,
        }


TrainingBatch = (
    ContrastiveBatch | KnowledgeDistillationBatch | MLMBatch | JointContrastiveMLMBatch
)


@dataclass(frozen=True)
class _Objective:
    batch_type: type
    step: Callable[..., Any]
    defaults: Mapping[str, float | int]
    metrics: tuple[str, ...]
    requires_mlm: bool = False


_OBJECTIVES = {
    "contrastive": _Objective(
        ContrastiveBatch, contrastive_train_step, {"temperature": 0.05}, ("loss",)
    ),
    "knowledge_distillation": _Objective(
        KnowledgeDistillationBatch,
        knowledge_distillation_train_step,
        {
            "temperature": 0.05,
            "kd_temperature": 0.07,
            "kd_alpha": 0.25,
            "false_negative_margin": 0.10,
            "teacher_confirmation_margin": 0.05,
            "duplicate_similarity_threshold": 0.999999,
        },
        ("loss", "contrastive_loss", "kd_loss"),
    ),
    "masked_language_modeling": _Objective(
        MLMBatch,
        masked_language_model_train_step,
        {"ignore_index": -100},
        ("loss",),
        requires_mlm=True,
    ),
    "joint_contrastive_mlm": _Objective(
        JointContrastiveMLMBatch,
        joint_contrastive_mlm_train_step,
        {
            "temperature": 0.05,
            "contrastive_weight": 1.0,
            "mlm_weight": 1.0,
            "ignore_index": -100,
        },
        ("loss", "contrastive_loss", "mlm_loss"),
        requires_mlm=True,
    ),
}


def _positive_integer(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    if value < 1:
        raise ValueError(f"{name} must be at least one.")


def _validate_options(options: Mapping[str, Any]) -> None:
    for name, value in options.items():
        if name == "ignore_index":
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError("ignore_index must be an integer.")
            continue
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{name} must be a real number.")
        if not isfinite(value):
            raise ValueError(f"{name} must be finite.")
        if name in {"temperature", "kd_temperature"} and value <= 0:
            raise ValueError(f"{name} must be greater than zero.")
        if name == "kd_alpha" and not 0 <= value <= 1:
            raise ValueError("kd_alpha must be in the range [0, 1].")
        if name == "duplicate_similarity_threshold" and not -1 <= value <= 1:
            raise ValueError("duplicate_similarity_threshold must be in the range [-1, 1].")
        if name.endswith(("_margin", "_weight")) and value < 0:
            raise ValueError(f"{name} must be nonnegative.")
    if (
        "mlm_weight" in options
        and options["mlm_weight"] + options["contrastive_weight"] == 0
    ):
        raise ValueError("At least one joint objective weight must be greater than zero.")


def _dataset_iterator(dataset: Any) -> Iterator[TrainingBatch]:
    if isinstance(dataset, (str, bytes, Mapping)):
        raise TypeError("train_dataset must be an iterable of training batches.")
    try:
        return iter(dataset)
    except TypeError as error:
        raise TypeError("train_dataset must be an iterable of training batches.") from error


def _require_interface(
    method: Callable[..., Any], name: str, *, training_mode: bool = False
) -> None:
    kwargs = {"deterministic": False} if training_mode else {}
    try:
        method_signature = signature(method)
        method_signature.bind(None, None, **kwargs)
        method_signature.bind(input_ids=None, attention_mask=None, **kwargs)
    except (TypeError, ValueError) as error:
        suffix = " and deterministic=False" if training_mode else ""
        raise TypeError(
            f"{name} must accept input_ids, attention_mask{suffix}."
        ) from error


class EmbeddingTrainer:
    """Train any supported encoder/objective through the same ``train()`` API.

    Configure ``train_dataset`` and optionally ``num_steps`` at construction,
    or pass batches and a step limit to ``train``. Without a step limit, one
    pass over the iterable is performed. Infinite iterables need a step limit.
    """

    def __init__(
        self,
        model: nnx.Module,
        optimizer: nnx.Optimizer,
        *,
        objective: str = "contrastive",
        train_dataset: Iterable[TrainingBatch] | None = None,
        num_steps: int | None = None,
        mesh: Mesh | None = None,
        temperature: float | None = None,
        objective_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(model, nnx.Module) or not callable(model):
            raise TypeError("model must be a callable Flax NNX Module.")
        if not isinstance(optimizer, nnx.Optimizer):
            raise TypeError(
                "optimizer must be a Flax NNX Optimizer, e.g. create_optimizer(model)."
            )
        if not isinstance(objective, str):
            raise TypeError("objective must be a string.")
        objective = "masked_language_modeling" if objective == "mlm" else objective
        if objective not in _OBJECTIVES:
            raise ValueError(
                f"Unknown objective {objective!r}. Choose from {tuple(_OBJECTIVES)}."
            )
        spec = _OBJECTIVES[objective]
        if spec.requires_mlm and not callable(
            getattr(model, "masked_language_model_logits", None)
        ):
            raise ValueError(
                f"{type(model).__name__} does not support objective {objective!r}: "
                "masked_language_model_logits is required. Use an MLM-capable model "
                "such as XLMRobertaEmbedding. Qwen3Embedding supports only contrastive "
                "and knowledge_distillation objectives."
            )
        _require_interface(
            model.__call__,
            "model.__call__",
            training_mode=objective == "joint_contrastive_mlm",
        )
        if spec.requires_mlm:
            _require_interface(
                model.masked_language_model_logits,
                "model.masked_language_model_logits",
                training_mode=True,
            )
        if mesh is not None:
            if not isinstance(mesh, Mesh):
                raise TypeError("mesh must be a JAX Mesh or None.")
            if len(mesh.axis_names) != 1:
                raise ValueError("EmbeddingTrainer requires a one-axis mesh.")
        model_mesh = get_model_mesh(model)
        if mesh is None:
            mesh = model_mesh
        elif model_mesh is not None and mesh != model_mesh:
            raise ValueError("mesh must match the model parameter mesh.")
        if mesh is not None and len(mesh.axis_names) != 1:
            raise ValueError("EmbeddingTrainer requires a one-axis mesh.")
        if num_steps is not None:
            _positive_integer("num_steps", num_steps)
        if train_dataset is not None:
            _dataset_iterator(train_dataset)
        if objective_kwargs is not None and not isinstance(objective_kwargs, Mapping):
            raise TypeError("objective_kwargs must be a mapping of objective parameters.")
        supplied_options = dict(objective_kwargs or {})
        if temperature is not None:
            if "temperature" in supplied_options:
                raise ValueError("Pass temperature directly or in objective_kwargs, not both.")
            supplied_options["temperature"] = temperature
        unknown = supplied_options.keys() - spec.defaults.keys()
        if unknown:
            raise ValueError(
                f"Unsupported parameters for objective {objective!r}: {sorted(unknown, key=str)}. "
                f"Allowed parameters: {tuple(spec.defaults)}."
            )
        options = {**spec.defaults, **supplied_options}
        _validate_options(options)
        self.model = model
        self.optimizer = optimizer
        self.objective = objective
        self.train_dataset = train_dataset
        self.num_steps = num_steps
        self.mesh = mesh
        self.temperature = options.get("temperature")
        self.objective_kwargs = options
        self._objective = spec

    def _train_batch(self, batch: TrainingBatch) -> dict[str, float]:
        if not isinstance(batch, self._objective.batch_type):
            raise TypeError(
                f"Objective {self.objective!r} requires {self._objective.batch_type.__name__}; "
                f"received {type(batch).__name__}."
            )
        inputs = batch._step_inputs()
        if self.mesh is not None:
            for name, value in inputs.items():
                if value.shape[0] % self.mesh.size:
                    raise ValueError(
                        f"{name} batch size {value.shape[0]} must be divisible by "
                        f"the mesh device count ({self.mesh.size})."
                    )
        inputs = {name: shard_batch(value, self.mesh) for name, value in inputs.items()}
        with mesh_context(self.mesh):
            result = self._objective.step(
                self.model, self.optimizer, **inputs, **self.objective_kwargs
            )
        values = result if isinstance(result, tuple) else (result,)
        return dict(
            zip(self._objective.metrics, map(float, jax.device_get(values)), strict=True)
        )

    def train_batch(self, batch: TrainingBatch) -> float:
        """Run one update for the configured objective and return its total loss."""
        return self._train_batch(batch)["loss"]

    def train(
        self,
        batches: Iterable[TrainingBatch] | None = None,
        *,
        num_steps: int | None = None,
    ) -> list[dict[str, float]]:
        """Own the training loop and return per-step host loss dictionaries.

        Every objective returns ``loss``. Distillation also returns
        ``contrastive_loss`` and ``kd_loss``; joint training additionally
        reports ``contrastive_loss`` and ``mlm_loss``. An explicit step limit
        overrides the constructor's limit; premature exhaustion raises an error.
        """
        dataset = self.train_dataset if batches is None else batches
        if dataset is None:
            raise ValueError("Provide train_dataset to the trainer or batches to train().")
        steps = self.num_steps if num_steps is None else num_steps
        if steps is not None:
            _positive_integer("num_steps", steps)
        iterator = _dataset_iterator(dataset)
        history = []
        while steps is None or len(history) < steps:
            try:
                batch = next(iterator)
            except StopIteration as error:
                if steps is not None:
                    raise ValueError(
                        f"batches ended after {len(history)} steps before num_steps={steps} "
                        "were completed. Supply enough batches or reduce num_steps."
                    ) from error
                break
            history.append(self._train_batch(batch))
        if not history:
            raise ValueError("train_dataset must contain at least one batch.")
        return history

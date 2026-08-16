"""High-level training helpers for tied encoder embedding models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import jax
from flax import nnx

from embex.training import contrastive_train_step
from embex.utils.distributed import shard_batch


@dataclass(frozen=True)
class ContrastiveBatch:
    """A tokenized batch of aligned query and positive-document pairs.

    All arrays retain the tokenizer's ``(batch, sequence)`` layout. Attention
    masks are required so every model call has explicit padding information.
    """

    query_input_ids: Any
    key_input_ids: Any
    query_attention_mask: Any
    key_attention_mask: Any

    def __post_init__(self) -> None:
        pairs = (
            ("query", self.query_input_ids, self.query_attention_mask),
            ("key", self.key_input_ids, self.key_attention_mask),
        )
        for name, input_ids, attention_mask in pairs:
            if getattr(input_ids, "shape", None) != getattr(attention_mask, "shape", None):
                raise ValueError(
                    f"{name}_input_ids and {name}_attention_mask must have the same shape."
                )
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
    ) -> "ContrastiveBatch":
        """Tokenize aligned text pairs with a Hugging Face tokenizer.

        ``return_tensors='np'`` deliberately keeps tokenization outside JAX and
        produces arrays that can be efficiently placed by the trainer.
        """
        if len(queries) != len(keys):
            raise ValueError("queries and keys must have equal lengths.")
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


class EmbeddingTrainer:
    """Minimal high-level trainer for paired embedding-model fine-tuning."""

    def __init__(
        self,
        model: nnx.Module,
        optimizer: nnx.Optimizer,
        *,
        mesh: Any = None,
        temperature: float = 0.05,
    ) -> None:
        if temperature <= 0:
            raise ValueError("temperature must be greater than zero.")
        self.model = model
        self.optimizer = optimizer
        self.mesh = mesh
        self.temperature = temperature

    def train_batch(self, batch: ContrastiveBatch) -> float:
        """Run one InfoNCE update and return a host scalar for logging."""
        loss = contrastive_train_step(
            self.model,
            self.optimizer,
            shard_batch(batch.query_input_ids, self.mesh),
            shard_batch(batch.key_input_ids, self.mesh),
            shard_batch(batch.query_attention_mask, self.mesh),
            shard_batch(batch.key_attention_mask, self.mesh),
            temperature=self.temperature,
        )
        return float(jax.device_get(loss))

    def fit(self, batches: Iterable[ContrastiveBatch], *, num_steps: int) -> list[float]:
        """Train for ``num_steps`` over an iterable of tokenized batches."""
        if num_steps < 1:
            raise ValueError("num_steps must be at least one.")
        iterator = iter(batches)
        losses = []
        for _ in range(num_steps):
            try:
                batch = next(iterator)
            except StopIteration as error:
                raise ValueError("batches ended before num_steps were completed.") from error
            losses.append(self.train_batch(batch))
        return losses

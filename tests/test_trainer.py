"""Integration and validation coverage for the common trainer API."""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from embex import (
    ContrastiveBatch,
    EmbeddingTrainer,
    JointContrastiveMLMBatch,
    KnowledgeDistillationBatch,
    MLMBatch,
)
from embex.models.qwen3_embedding import Qwen3EmbeddingConfig, create_qwen3_embedding
from embex.models.xlm_roberta import XLMRobertaConfig, create_xlm_roberta
from embex.training import (
    contrastive_train_step,
    create_optimizer,
    joint_contrastive_mlm_train_step,
    knowledge_distillation_train_step,
    masked_language_model_train_step,
)


def make_model(family="xlm_roberta", dtype=jnp.float32):
    common = dict(
        vocab_size=32, hidden_size=8, intermediate_size=16,
        num_hidden_layers=1, num_attention_heads=2, dtype=dtype,
    )
    if family == "qwen3":
        return create_qwen3_embedding(
            Qwen3EmbeddingConfig(**common, head_dim=4, num_key_value_heads=1), seed=3,
        )[0]
    return create_xlm_roberta(
        XLMRobertaConfig(**common, max_position_embeddings=16), seed=3,
    )[0]


@pytest.fixture
def batches():
    query_ids = np.array([[0, 4, 5, 2], [0, 6, 2, 1]], dtype=np.int32)
    key_ids = np.array([[0, 7, 2], [0, 8, 2]], dtype=np.int32)
    pairs = ContrastiveBatch(query_ids, key_ids, (query_ids != 1), (key_ids != 1))
    # MLM need not have the same batch size or sequence length as the pairs.
    mlm = MLMBatch(
        np.array([[0, 9, 2]], dtype=np.int32),
        np.ones((1, 3), dtype=np.int32),
        np.array([[-1, 4, -1]], dtype=np.int32),
    )
    kd = KnowledgeDistillationBatch(
        pairs,
        np.array([[1., 0., 0.], [0., 1., 0.]], dtype=np.float32),
        np.array([[0.9, 0.1, 0.], [0.1, 0.9, 0.]], dtype=np.float32),
    )
    return {
        "contrastive": pairs,
        "knowledge_distillation": kd,
        "masked_language_modeling": mlm,
        "joint_contrastive_mlm": JointContrastiveMLMBatch(pairs, mlm),
    }


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16], ids=["float32", "bfloat16"])
@pytest.mark.parametrize("family,objective", [
    ("qwen3", "contrastive"),
    ("qwen3", "knowledge_distillation"),
    ("xlm_roberta", "contrastive"),
    ("xlm_roberta", "knowledge_distillation"),
    ("xlm_roberta", "masked_language_modeling"),
    ("xlm_roberta", "joint_contrastive_mlm"),
])
def test_train_matches_direct_updates_for_every_supported_combination(family, objective, dtype, batches):
    model = make_model(family, dtype)
    reference = make_model(family, dtype)
    optimizer = create_optimizer(model, 1e-3)
    reference_optimizer = create_optimizer(reference, 1e-3)
    options = {
        "contrastive": {"temperature": 0.2},
        "knowledge_distillation": {
            "temperature": 0.3, "kd_temperature": 0.4, "kd_alpha": 0.6,
            "false_negative_margin": 0.2, "teacher_confirmation_margin": 0.1,
            "duplicate_similarity_threshold": 0.99,
        },
        "masked_language_modeling": {"ignore_index": -1},
        "joint_contrastive_mlm": {
            "temperature": 0.2, "contrastive_weight": 0.3,
            "mlm_weight": 0.7, "ignore_index": -1,
        },
    }[objective]
    trainer = EmbeddingTrainer(
        model, optimizer, objective=objective,
        train_dataset=iter([batches[objective]] * 3), num_steps=2,
        objective_kwargs=options,
    )
    before = [np.array(value) for value in jax.tree.leaves(nnx.state(model, nnx.Param))]
    history = trainer.train()

    pairs = batches["contrastive"]
    pair_args = (
        pairs.query_input_ids, pairs.key_input_ids,
        pairs.query_attention_mask, pairs.key_attention_mask,
    )
    mlm = batches["masked_language_modeling"]
    mlm_args = (mlm.input_ids, mlm.attention_mask, mlm.labels)
    kd = batches["knowledge_distillation"]
    for metrics in history:
        if objective == "contrastive":
            result = (contrastive_train_step(
                reference, reference_optimizer, *pair_args, **options,
            ),)
            names = ("loss",)
        elif objective == "knowledge_distillation":
            result = knowledge_distillation_train_step(
                reference, reference_optimizer, *pair_args,
                kd.teacher_query_embeddings, kd.teacher_key_embeddings, **options,
            )
            names = ("loss", "contrastive_loss", "kd_loss")
        elif objective == "masked_language_modeling":
            result = (masked_language_model_train_step(
                reference, reference_optimizer, *mlm_args, **options,
            ),)
            names = ("loss",)
        else:
            result = joint_contrastive_mlm_train_step(
                reference, reference_optimizer, *pair_args, *mlm_args, **options,
            )
            names = ("loss", "contrastive_loss", "mlm_loss")
        assert set(metrics) == set(names)
        for name, expected in zip(names, result, strict=True):
            assert isinstance(metrics[name], float)
            assert np.isfinite(metrics[name])
            np.testing.assert_allclose(metrics[name], expected, rtol=1e-5, atol=1e-6)

    assert len(history) == 2
    assert int(optimizer.step.get_value()) == 2
    after = jax.tree.leaves(nnx.state(model, nnx.Param))
    assert any(not np.array_equal(a, b) for a, b in zip(before, after, strict=True))
    for actual, expected in zip(
        after, jax.tree.leaves(nnx.state(reference, nnx.Param)), strict=True,
    ):
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.fixture(scope="module")
def validation_model():
    model = make_model()
    return model, create_optimizer(model)


@pytest.mark.parametrize("objective", ["mlm", "masked_language_modeling", "joint_contrastive_mlm"])
def test_qwen_rejects_mlm_objectives_at_construction(objective):
    model = make_model("qwen3")
    with pytest.raises(ValueError, match="Qwen3Embedding.*does not support"):
        EmbeddingTrainer(model, create_optimizer(model), objective=objective)


@pytest.mark.parametrize("kwargs,error,match", [
    ({"objective": "unknown"}, ValueError, "Unknown objective"),
    ({"objective": None}, TypeError, "objective must be a string"),
    ({"temperature": 0}, ValueError, "temperature"),
    ({"temperature": float("nan")}, ValueError, "finite"),
    ({"temperature": float("inf")}, ValueError, "finite"),
    ({"temperature": "warm"}, TypeError, "real number"),
    ({"temperature": True}, TypeError, "real number"),
    ({"num_steps": 0}, ValueError, "num_steps"),
    ({"num_steps": 1.5}, TypeError, "num_steps"),
    ({"num_steps": True}, TypeError, "num_steps"),
    ({"train_dataset": 4}, TypeError, "iterable"),
    ({"train_dataset": "data"}, TypeError, "iterable"),
    ({"train_dataset": {}}, TypeError, "iterable"),
    ({"mesh": "fsdp"}, TypeError, "JAX Mesh"),
    ({"objective_kwargs": []}, TypeError, "mapping"),
    ({"objective_kwargs": {"kd_alpha": 0.5}}, ValueError, "Unsupported parameters"),
    ({"temperature": 0.1, "objective_kwargs": {"temperature": 0.2}}, ValueError, "not both"),
    ({"objective": "mlm", "temperature": 0.1}, ValueError, "Unsupported parameters"),
    ({"objective": "mlm", "objective_kwargs": {"ignore_index": 1.5}}, TypeError, "integer"),
    ({"objective": "knowledge_distillation", "objective_kwargs": {"kd_alpha": 1.1}}, ValueError, "kd_alpha"),
    ({"objective": "knowledge_distillation", "objective_kwargs": {"kd_temperature": -1}}, ValueError, "kd_temperature"),
    ({"objective": "knowledge_distillation", "objective_kwargs": {"false_negative_margin": -1}}, ValueError, "nonnegative"),
    ({"objective": "knowledge_distillation", "objective_kwargs": {"duplicate_similarity_threshold": 2}}, ValueError, "range"),
    ({"objective": "joint_contrastive_mlm", "objective_kwargs": {"mlm_weight": -1}}, ValueError, "nonnegative"),
    ({"objective": "joint_contrastive_mlm", "objective_kwargs": {"mlm_weight": 0, "contrastive_weight": 0}}, ValueError, "At least one"),
])
def test_invalid_trainer_configuration_is_rejected(validation_model, kwargs, error, match):
    model, optimizer = validation_model
    with pytest.raises(error, match=match):
        EmbeddingTrainer(model, optimizer, **kwargs)


def test_invalid_model_and_optimizer_are_rejected(validation_model):
    model, optimizer = validation_model
    with pytest.raises(TypeError, match="model must be"):
        EmbeddingTrainer(object(), optimizer)
    with pytest.raises(TypeError, match="optimizer must be"):
        EmbeddingTrainer(model, object())
    linear = nnx.Linear(8, 8, rngs=nnx.Rngs(0))
    with pytest.raises(TypeError, match="must accept input_ids, attention_mask"):
        EmbeddingTrainer(linear, create_optimizer(linear))

    class InvalidMLM(nnx.Module):
        def __call__(self, input_ids, attention_mask=None):
            return input_ids

        def masked_language_model_logits(self, input_ids, attention_mask=None):
            return input_ids

    invalid_mlm = InvalidMLM()
    with pytest.raises(TypeError, match="deterministic=False"):
        EmbeddingTrainer(invalid_mlm, create_optimizer(invalid_mlm), objective="mlm")


@pytest.mark.parametrize("objective", [
    "contrastive", "knowledge_distillation", "masked_language_modeling", "joint_contrastive_mlm",
])
def test_wrong_batch_is_rejected_before_optimizer_update(validation_model, objective):
    model, optimizer = validation_model
    trainer = EmbeddingTrainer(model, optimizer, objective=objective, train_dataset=[{}])
    step = int(optimizer.step.get_value())
    with pytest.raises(TypeError, match="requires.*received dict"):
        trainer.train()
    assert int(optimizer.step.get_value()) == step


def test_train_loop_limits_overrides_and_single_batch_updates(batches):
    model = make_model("qwen3")
    batch = batches["contrastive"]
    trainer = EmbeddingTrainer(
        model, create_optimizer(model), train_dataset=[batch], num_steps=5, temperature=0.2,
    )
    assert trainer.temperature == 0.2
    assert len(trainer.train(num_steps=1)) == 1
    assert len(trainer.train(itertools.repeat(batch), num_steps=2)) == 2
    assert isinstance(trainer.train_batch(batch), float)
    history = trainer.train([batch, batch], num_steps=2)
    losses = [metrics["loss"] for metrics in history]
    assert len(losses) == 2 and all(isinstance(loss, float) for loss in losses)
    with pytest.raises(ValueError, match="ended after 1 steps"):
        trainer.train([batch], num_steps=2)
    for invalid, error in [(0, ValueError), (-1, ValueError), (1.5, TypeError), (True, TypeError)]:
        with pytest.raises(error, match="num_steps"):
            trainer.train(num_steps=invalid)


def test_train_one_pass_empty_missing_and_one_shot_data(batches):
    model = make_model("qwen3")
    trainer = EmbeddingTrainer(model, create_optimizer(model))
    with pytest.raises(ValueError, match="Provide train_dataset"):
        trainer.train()
    with pytest.raises(ValueError, match="at least one batch"):
        trainer.train([])
    assert len(trainer.train(iter([batches["contrastive"]] * 2))) == 2
    trainer.train_dataset = iter([batches["contrastive"]])
    assert len(trainer.train()) == 1
    with pytest.raises(ValueError, match="at least one batch"):
        trainer.train()


def test_mlm_alias_and_batch_validation(batches, validation_model):
    model, optimizer = validation_model
    trainer = EmbeddingTrainer(model, optimizer, objective="mlm")
    assert trainer.objective == "masked_language_modeling"
    pairs = batches["contrastive"]
    ids, mask = pairs.query_input_ids, pairs.query_attention_mask
    with pytest.raises(ValueError, match="two-dimensional"):
        ContrastiveBatch(ids[0], ids, mask[0], mask)
    with pytest.raises(ValueError, match="nonempty"):
        ContrastiveBatch(ids[:0], ids[:0], mask[:0], mask[:0])
    with pytest.raises(ValueError, match="same shape"):
        ContrastiveBatch(ids, ids, mask[:, :1], mask)
    with pytest.raises(ValueError, match="same number of rows"):
        ContrastiveBatch(ids, ids[:1], mask, mask[:1])
    with pytest.raises(TypeError, match="integer"):
        ContrastiveBatch(ids.astype(float), ids, mask, mask)
    with pytest.raises(TypeError, match="numeric or boolean"):
        ContrastiveBatch(ids, ids, mask.astype(complex), mask)
    with pytest.raises(ValueError, match="same shape"):
        MLMBatch(ids, mask, ids[:, :1])
    with pytest.raises(TypeError, match="integer"):
        MLMBatch(ids, mask, ids.astype(float))
    with pytest.raises(ValueError, match="same row count"):
        KnowledgeDistillationBatch(pairs, np.ones((1, 3)), np.ones((1, 3)))
    with pytest.raises(ValueError, match="same shape"):
        KnowledgeDistillationBatch(pairs, np.ones((2, 3)), np.ones((2, 4)))
    with pytest.raises(TypeError, match="floating-point"):
        KnowledgeDistillationBatch(pairs, ids, ids)
    with pytest.raises(TypeError, match="ContrastiveBatch"):
        KnowledgeDistillationBatch(None, np.ones((2, 3)), np.ones((2, 3)))
    with pytest.raises(TypeError, match="ContrastiveBatch"):
        JointContrastiveMLMBatch(None, batches["masked_language_modeling"])
    with pytest.raises(TypeError, match="MLMBatch"):
        JointContrastiveMLMBatch(pairs, None)


def test_tokenizer_batches_work_with_train():
    class Tokenizer:
        def __call__(self, texts, **kwargs):
            assert kwargs == {
                "padding": "max_length", "truncation": True,
                "max_length": 3, "return_tensors": "np",
            }
            ids = np.array([[0, len(text) + 3, 2] for text in texts], dtype=np.int32)
            return {"input_ids": ids, "attention_mask": np.ones_like(ids)}

    batch = ContrastiveBatch.from_tokenizer(
        Tokenizer(), ["a", "bb"], ["ccc", "dddd"], max_length=3,
    )
    model = make_model("qwen3")
    trainer = EmbeddingTrainer(model, create_optimizer(model), train_dataset=[batch])
    assert np.isfinite(trainer.train()[0]["loss"])
    with pytest.raises(ValueError, match="equal lengths"):
        ContrastiveBatch.from_tokenizer(Tokenizer(), ["a"], [])
    with pytest.raises(ValueError, match="must not be empty"):
        ContrastiveBatch.from_tokenizer(Tokenizer(), [], [])
    with pytest.raises(ValueError, match="max_length"):
        ContrastiveBatch.from_tokenizer(Tokenizer(), ["a"], ["a"], max_length=0)


def test_compatible_custom_nnx_encoder_can_use_the_trainer(batches):
    class Encoder(nnx.Module):
        def __init__(self):
            self.embedding = nnx.Embed(32, 8, rngs=nnx.Rngs(0))

        def __call__(self, input_ids, attention_mask=None):
            states = self.embedding(input_ids)
            mask = attention_mask[..., None]
            return (states * mask).sum(axis=1) / mask.sum(axis=1)

    model = Encoder()
    trainer = EmbeddingTrainer(
        model, create_optimizer(model), train_dataset=[batches["contrastive"]],
    )
    assert np.isfinite(trainer.train()[0]["loss"])

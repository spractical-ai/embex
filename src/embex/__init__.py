"""EmbeX: JAX/Flax NNX components for embedding-model training."""

from embex.training import (
    contrastive_train_step,
    create_optimizer,
    joint_contrastive_mlm_train_step,
    knowledge_distillation_train_step,
    masked_language_model_train_step,
)
from embex.trainer import ContrastiveBatch, EmbeddingTrainer
from embex.utils.loss_functions import (
    infonce_kd_loss,
    infonce_loss,
    masked_language_model_loss,
)

__all__ = [
    "ContrastiveBatch",
    "EmbeddingTrainer",
    "contrastive_train_step",
    "create_optimizer",
    "infonce_kd_loss",
    "infonce_loss",
    "joint_contrastive_mlm_train_step",
    "knowledge_distillation_train_step",
    "masked_language_model_loss",
    "masked_language_model_train_step",
]

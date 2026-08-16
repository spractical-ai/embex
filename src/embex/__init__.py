"""EmbeX: JAX/Flax NNX components for embedding-model training."""

from embex.training import create_optimizer, contrastive_train_step
from embex.trainer import ContrastiveBatch, EmbeddingTrainer
from embex.utils.loss_functions import infonce_loss

__all__ = [
    "ContrastiveBatch",
    "EmbeddingTrainer",
    "contrastive_train_step",
    "create_optimizer",
    "infonce_loss",
]

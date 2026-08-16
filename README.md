# EmbeX

EmbeX is a small JAX and Flax NNX library for training embedding models. It
currently implements Qwen3-Embedding and one-directional InfoNCE, preserving
the model and training behavior from the original notebook while moving reusable
pieces into a package.

## Install

### From GitHub with uv

Add EmbeX to another uv project directly from GitHub:

```bash
uv add "embex @ git+https://github.com/spractical-ai/embex.git"
```

Or install it into an existing uv environment without changing that project's
`pyproject.toml`:

```bash
uv pip install "embex @ git+https://github.com/spractical-ai/embex.git"
```

### From a clone

```bash
git clone https://github.com/spractical-ai/embex.git
cd embex
uv sync
```

`uv sync` creates `.venv`, installs the package in editable mode, and uses the
committed `uv.lock` for reproducible installs. EmbeX requires Python 3.12 and
pins JAX 0.9.2 and Flax 0.12.6. Select the accelerator-specific sync command
below before running training.

## Accelerator installation

The EmbeX code is accelerator-agnostic: it discovers JAX devices at runtime and
uses the same model and mesh code for CPU, NVIDIA GPU, or Google Cloud TPU.
The JAX binary is not universal, however, so select **exactly one** backend for
the machine on which you train:

| Hardware | Clone install | Add EmbeX to another uv project |
| --- | --- | --- |
| CPU | `uv sync` | `uv add "embex @ git+https://github.com/spractical-ai/embex.git"` |
| NVIDIA GPU on Linux, CUDA 12 | `uv sync --group cuda12` | `uv add "embex @ git+https://github.com/spractical-ai/embex.git" "jax[cuda12]==0.9.2"` |
| NVIDIA GPU on Linux, CUDA 13 | `uv sync --group cuda13` | `uv add "embex @ git+https://github.com/spractical-ai/embex.git" "jax[cuda13]==0.9.2"` |
| Google Cloud TPU VM (Linux) | `uv sync --group tpu` | `uv add "embex @ git+https://github.com/spractical-ai/embex.git" "jax[tpu]==0.9.2"` |

The CUDA groups use the JAX pip-distributed CUDA and cuDNN runtime. For a
locally installed CUDA toolkit, AMD ROCm, Intel GPU, or a platform outside this
table, follow the matching JAX installation instructions, pin JAX to `0.9.2`,
then install EmbeX normally. Verify the selected backend with:

```bash
uv run python -c "import jax; print(jax.default_backend(), jax.devices())"
```

For development dependencies:

```bash
uv sync --group dev
```

## Minimal Qwen3 training setup

```python
import jax
import optax

from embex.models.qwen3_embedding import (
    Qwen3EmbeddingConfig,
    create_qwen3_embedding,
    load_qwen3_embedding_weights,
)
from embex.training import contrastive_train_step, create_optimizer
from embex.utils.distributed import shard_batch

config = Qwen3EmbeddingConfig.from_preset("0.6B")
model, mesh = create_qwen3_embedding(config)
load_qwen3_embedding_weights(model, "model.safetensors")

schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0,
    peak_value=2e-5,
    warmup_steps=200,
    decay_steps=2_000,
    end_value=0.0,
)
optimizer = create_optimizer(model, schedule)

# Tokenize a paired query/context batch outside the library.
queries = shard_batch(query_input_ids, mesh)
keys = shard_batch(context_input_ids, mesh)
query_masks = shard_batch(query_attention_mask, mesh)
key_masks = shard_batch(context_attention_mask, mesh)
loss = contrastive_train_step(model, optimizer, queries, keys, query_masks, key_masks)
```

`create_qwen3_embedding` uses standard NNX initializers on one device. With
multiple JAX devices it creates a one-axis FSDP mesh and applies the notebook's
parameter partitioning pattern. Pass a custom `mesh` and `partition_axis` to
control placement explicitly. The factory always constructs the model inside
`with jax.set_mesh(mesh):`; direct multi-device construction is rejected unless
you have entered a mesh context and supplied its partition axis yourself.

## Checkpoint conversion

```python
from embex.models.qwen3_embedding import qwen3_embedding_hf_state_dict
from embex.utils.save_model import save_model

save_model(model, "output/model.safetensors", qwen3_embedding_hf_state_dict)
```

For a complete Hugging Face-compatible directory, use
`export_huggingface_model` from `embex.utils.save_model` with the same state-dict
factory.

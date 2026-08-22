# EmbeX

EmbeX is a small JAX and Flax NNX library for training embedding models. It
supports Qwen3-Embedding and XLM-RoBERTa encoders together with InfoNCE,
relational knowledge distillation, and masked language modeling, preserving the
original Qwen notebook's behavior while moving reusable pieces into a package.

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

## Installation on Kaggle

`uv sync` creates a separate `.venv`, while a Kaggle notebook continues to use
its already-running kernel Python. Install EmbeX into that exact interpreter so
plain notebook cells can use `import embex`:

```python
!git clone https://github.com/spractical-ai/embex.git
%cd embex

import sys
!uv pip install --python {sys.executable} --reinstall -e .
!{sys.executable} -c "import embex; print(embex.__file__)"
```

The final command should print a path ending in
`embex/src/embex/__init__.py`. Restart the Kaggle session before importing from
EmbeX in normal Python cells; Python reads editable-install paths when the
kernel starts. This CPU command uses the default JAX dependency. For a GPU or
TPU notebook, use the corresponding accelerator command below instead.

For a terminal or script, keep the isolated environment and prefix commands
with `uv run`, for example `uv run python train.py`.

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

## Supported models

| Model | Encoder | Training objectives | Presets |
| --- | --- | --- | --- |
| Qwen3-Embedding | Decoder-only, causal attention with final-token pooling | InfoNCE, relational KD | 0.6B, 4B, 8B |
| XLM-RoBERTa | Bidirectional attention with masked-mean or first-token (`cls`) pooling | InfoNCE, relational KD, MLM, or contrastive + MLM | Base, large |

Both implementations expose the `(input_ids, attention_mask) -> embeddings`
interface used by `EmbeddingTrainer`, support single-device and FSDP model
creation, and include Hugging Face safetensors import/export adapters.
XLM-RoBERTa also exposes `encode_tokens` for contextual states and
`masked_language_model_logits` for token prediction through a decoder tied to
the input word embeddings. Its weight loader accepts both bare backbone keys
and common `roberta.`-prefixed checkpoints, loading a compatible `lm_head` when
one is present while ignoring unrelated task heads.

`masked_language_model_loss` averages token cross-entropy only where labels are
not `-100`. Use `masked_language_model_train_step` for MLM-only updates or
`joint_contrastive_mlm_train_step` for a weighted combination of InfoNCE and
MLM. These MLM APIs require XLM-RoBERTa; Qwen3-Embedding intentionally exposes
only the contrastive path.

The tutorials below intentionally remain focused on Qwen3-Embedding.

## Inference tutorial

EmbeX expects Hugging Face tokenizer outputs and **always passes an attention
mask with input IDs**. Use the exact same tokenization behavior for inference
and training; omitting the mask changes padding and pooling behavior.

```python
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from embex.models.qwen3_embedding import (
    Qwen3EmbeddingConfig,
    create_qwen3_embedding,
    load_qwen3_embedding_weights,
)
from embex.utils.distributed import shard_batch

config = Qwen3EmbeddingConfig.from_preset("0.6B")
model, mesh = create_qwen3_embedding(config)

weights_path = hf_hub_download(
    repo_id="Qwen/Qwen3-Embedding-0.6B", filename="model.safetensors"
)
load_qwen3_embedding_weights(model, weights_path)

tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen3-Embedding-0.6B", padding_side="left"
)
texts = [
    "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:capital of China",
    "The capital of China is Beijing.",
]
tokens = tokenizer(
    texts,
    padding="max_length",
    truncation=True,
    max_length=512,
    return_tensors="np",
)

# Both values come from the Hugging Face tokenizer and are passed to the model.
embeddings = model(
    shard_batch(tokens["input_ids"], mesh),
    shard_batch(tokens["attention_mask"], mesh),
)
print(embeddings.shape)  # (2, 1024) for the 0.6B preset
```

## Training tutorial

This example uses dummy text pairs and a tiny Qwen-shaped configuration so it
is safe to run on CPU. It demonstrates the high-level trainer API; it is not a
useful retrieval model. Train a pretrained 0.6B model by using
`Qwen3EmbeddingConfig.from_preset("0.6B")`, loading its weights as in the
inference tutorial, and supplying real aligned query/document pairs.

```python
import jax.numpy as jnp
import optax
from transformers import AutoTokenizer

from embex.models.qwen3_embedding import Qwen3EmbeddingConfig, create_qwen3_embedding
from embex.trainer import ContrastiveBatch, EmbeddingTrainer
from embex.training import create_optimizer

tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen3-Embedding-0.6B", padding_side="left"
)

# Small architecture for a quick CPU demonstration; it starts from random weights.
config = Qwen3EmbeddingConfig(
    vocab_size=len(tokenizer),
    hidden_size=64,
    head_dim=16,
    intermediate_size=128,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    dtype=jnp.float32,
)
model, mesh = create_qwen3_embedding(config)

schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0, peak_value=2e-4, warmup_steps=1, decay_steps=3, end_value=0.0
)
trainer = EmbeddingTrainer(model, create_optimizer(model, schedule), mesh=mesh)

batch = ContrastiveBatch.from_tokenizer(
    tokenizer,
    queries=[
        "Instruct: retrieve a relevant passage\nQuery:capital of China",
        "Instruct: retrieve a relevant passage\nQuery:what causes gravity",
    ],
    keys=[
        "Beijing is the capital of China.",
        "Gravity attracts bodies with mass.",
    ],
    max_length=32,
)

for step in range(3):
    print(f"step={step + 1}, loss={trainer.train_batch(batch):.4f}")
```

`ContrastiveBatch.from_tokenizer` requires a Hugging Face tokenizer and retains
both attention masks. `EmbeddingTrainer` places every input and mask on the
selected mesh, then calls the model with `(input_ids, attention_mask)` for both
the query and positive-document encoder passes.

`create_qwen3_embedding` uses standard NNX initializers on one device. With
multiple JAX devices it creates a one-axis FSDP mesh and applies the notebook's
parameter partitioning pattern. Pass a custom `mesh` and `partition_axis` to
control placement explicitly. The factory always constructs the model inside
`with jax.set_mesh(mesh):`; direct multi-device construction is rejected unless
you have entered a mesh context and supplied its partition axis yourself.

## Relational knowledge distillation

Use `knowledge_distillation_train_step` when teacher query and document
embeddings have already been computed. Teacher and student embedding widths do
not need to match. Rows must remain aligned across all four batches.

The underlying objective is exposed separately as `infonce_kd_loss`. The
original `infonce_loss` remains the unchanged, teacher-free contrastive loss.

```python
from embex.training import knowledge_distillation_train_step

total_loss, hard_loss, kd_loss = knowledge_distillation_train_step(
    model,
    optimizer,
    query_input_ids,
    key_input_ids,
    query_attention_mask,
    key_attention_mask,
    teacher_query_embeddings,
    teacher_key_embeddings,
    temperature=0.05,
    kd_temperature=0.07,
    kd_alpha=0.25,
)
```

The hard component is query-to-document InfoNCE. It masks an off-diagonal
candidate only when the student ranks it above the labeled positive by the
configured margin and the teacher confirms it, or when teacher document
embeddings identify duplicate contexts. The KD component retains every pair and
matches the teacher's query-to-document and document-to-query relational
distributions using the same temperature for student and teacher logits.

## Checkpoint conversion

```python
from embex.models.qwen3_embedding import qwen3_embedding_hf_state_dict
from embex.utils.save_model import save_model

save_model(model, "output/model.safetensors", qwen3_embedding_hf_state_dict)
```

For a complete Hugging Face-compatible directory, use
`export_huggingface_model` from `embex.utils.save_model` with the same state-dict
factory. XLM-RoBERTa provides the equivalent `xlm_roberta_hf_state_dict`
backbone factory and `xlm_roberta_mlm_hf_state_dict` masked-LM factory in
`embex.models.xlm_roberta`.

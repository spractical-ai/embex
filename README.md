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

The inference tutorial uses Qwen3-Embedding; the shared trainer below supports
both model families and every objective in the table.

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

trainer = EmbeddingTrainer(
    model,
    create_optimizer(model, schedule),
    mesh=mesh,
    objective="contrastive",
    train_dataset=[batch] * 3,
    num_steps=3,
)
history = trainer.train()
print(history)  # [{"loss": ...}, {"loss": ...}, {"loss": ...}]
```

`ContrastiveBatch.from_tokenizer` requires a Hugging Face tokenizer and retains
both attention masks. `EmbeddingTrainer` places every input and mask on the
selected mesh, then calls the model with `(input_ids, attention_mask)` for both
the query and positive-document encoder passes.
When `mesh` is omitted, the trainer infers it from the model's parameter
placement. `create_optimizer` initializes optimizer state in that same mesh
context, and the trainer activates it for each update.

### One trainer for every objective

Choose the objective when constructing `EmbeddingTrainer`. The same
`trainer.train()` call handles model updates and batch placement for each
supported combination:

| `objective` | Batch type | Qwen3-Embedding | XLM-RoBERTa |
| --- | --- | --- | --- |
| `"contrastive"` (default) | `ContrastiveBatch` | Yes | Yes |
| `"knowledge_distillation"` | `KnowledgeDistillationBatch` | Yes | Yes |
| `"masked_language_modeling"` (alias `"mlm"`) | `MLMBatch` | No | Yes |
| `"joint_contrastive_mlm"` | `JointContrastiveMLMBatch` | No | Yes |

`train_dataset` is an iterable of batches, such as a list, generator, or data
loader. With no `num_steps`, training makes one pass over that iterable. Set
`num_steps` for infinite streams or to stop after a fixed number of updates;
if the data ends early, the trainer raises `ValueError`. One-shot generators
are consumed by training; supply a new iterator for another run. Data is not
automatically repeated or shuffled. You can also supply data and a step limit
at call time: `trainer.train(batches, num_steps=100)`.

Every history entry contains a Python float `loss`. Distillation also reports
`contrastive_loss` (its false-negative-masked hard component) and `kd_loss`;
joint training reports `contrastive_loss` and `mlm_loss`.

For distillation, wrap tokenized pairs with precomputed teacher embeddings.
Teacher query/key rows must align with the tokenized pairs, but the teacher
embedding width may differ from the student width:

```python
from embex import KnowledgeDistillationBatch

kd_batch = KnowledgeDistillationBatch(
    contrastive=batch,
    teacher_query_embeddings=teacher_query_embeddings,
    teacher_key_embeddings=teacher_key_embeddings,
)
trainer = EmbeddingTrainer(
    model,
    create_optimizer(model),
    mesh=mesh,
    objective="knowledge_distillation",
    train_dataset=[kd_batch],
    objective_kwargs={"kd_alpha": 0.25, "kd_temperature": 0.07},
)
history = trainer.train()
```

For MLM, use an XLM-RoBERTa model. Your data pipeline supplies already masked
token IDs and target labels, using `-100` at positions that should not contribute
to the loss. The trainer does not select or corrupt tokens automatically:

```python
from embex import JointContrastiveMLMBatch, MLMBatch
from embex.models.xlm_roberta import XLMRobertaConfig, create_xlm_roberta

xlmr, xlmr_mesh = create_xlm_roberta(XLMRobertaConfig.from_preset("base"))
mlm_batch = MLMBatch(masked_input_ids, attention_mask, labels)
trainer = EmbeddingTrainer(
    xlmr,
    create_optimizer(xlmr),
    mesh=xlmr_mesh,
    objective="masked_language_modeling",
    train_dataset=[mlm_batch],
)
history = trainer.train()

# Tokenize these pairs with the XLM-RoBERTa tokenizer as well.
joint_batch = JointContrastiveMLMBatch(contrastive=xlmr_pairs, mlm=mlm_batch)
trainer = EmbeddingTrainer(
    xlmr,
    create_optimizer(xlmr),
    mesh=xlmr_mesh,
    objective="joint_contrastive_mlm",
    train_dataset=[joint_batch],
    objective_kwargs={"contrastive_weight": 1.0, "mlm_weight": 0.1},
)
history = trainer.train()
```

Use `objective_kwargs` to override the low-level step's parameters:

| Objective | Options and defaults |
| --- | --- |
| Contrastive | `temperature=0.05` |
| Distillation | `temperature=0.05`, `kd_temperature=0.07`, `kd_alpha=0.25`, `false_negative_margin=0.10`, `teacher_confirmation_margin=0.05`, `duplicate_similarity_threshold=0.999999` |
| MLM | `ignore_index=-100` |
| Joint contrastive/MLM | `temperature=0.05`, `contrastive_weight=1.0`, `mlm_weight=1.0`, `ignore_index=-100` |

The existing `temperature=` constructor argument is also supported for
contrastive, distillation, and joint training; specify it in only one place.
Unknown objectives, irrelevant options, invalid numerical settings, and
incompatible models raise errors at construction. For example, choosing MLM
with Qwen3-Embedding raises `ValueError` before any training runs. Model
compatibility is checked through its callable interfaces, allowing future NNX
encoders to reuse the trainer. Malformed batch shapes/dtypes are rejected by
the batch constructors; the trainer checks objective/batch compatibility and
mesh batch divisibility before dispatching an update.

`train()` is the shared training-loop API. `train_batch(batch)` runs a single
update and returns its total loss. To extract only total losses from a training
run, use `[step["loss"] for step in trainer.train()]`. The low-level training
steps remain available for custom loops.

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

## Testing

Install the development dependencies, then run the complete suite from the
repository root:

```bash
uv sync --group dev
uv run pytest -q
```

To run just the shared trainer checks:

```bash
uv run pytest -q tests/test_trainer.py tests/test_trainer_distributed.py
```

The tests use tiny, randomly initialized models and synthetic token arrays.
They do not download pretrained checkpoints or datasets. Trainer/reference
comparisons cover float32 and the default bfloat16 compute precision. Fixed
seeds make those comparisons repeatable, including XLM-RoBERTa's dropout.

| Test file | What it verifies |
| --- | --- |
| `tests/test_trainer.py` | All six supported model/objective combinations, objective options, returned metrics, real parameter updates, constructor and batch validation, iterator exhaustion, step limits, and compatible custom NNX encoders. |
| `tests/test_trainer_distributed.py` | Training for all six combinations on two virtual CPU devices, inferred and explicit meshes, retained parameter sharding, and rejection of incompatible meshes and batch sizes. |
| `tests/test_knowledge_distillation.py` | Plain InfoNCE, duplicate-context masking, the weighted KD objective, stopped teacher gradients, and finite training losses. |
| `tests/test_masked_language_modeling.py` | Ignored labels and zero-target loss, MLM checkpoint round trips, and MLM/joint updates. |
| `tests/test_xlm_roberta.py` | Padding-aware position IDs and pooling, token states, dropout, backbone weight round trips, and architecture presets. |
| `tests/test_tiny_model_inference.py` | Output shape, dtype, finiteness, and repeatability for Qwen embeddings, XLM-R embeddings, and MLM logits; also reports compilation and cached-call timings. |
| `tests/test_tiny_model_inference_8_devices.py` | Automatic mesh creation and sharded inference on eight virtual CPU devices. |

For each trainer/model/objective combination, the integration test creates two
identical models. One receives two updates through `trainer.train()`; the other
receives the same updates through the corresponding low-level training step.
It compares the per-step losses and final parameters, checks the optimizer
step count, and verifies that parameters actually changed. Non-default objective
options ensure the trainer forwards the settings correctly.

Invalid-input tests check both the exception type and its message. Rejected
batch and mesh-divisibility checks also verify that the optimizer did not
advance before the error.

The distributed tests start separate Python processes with
`JAX_PLATFORMS=cpu` and an `XLA_FLAGS` device-count setting. JAX reads that count
when its backend starts, so subprocess isolation keeps the forced two/eight
device counts from affecting the rest of the suite. These checks exercise
sharding behavior on CPU; they do not establish real GPU/TPU performance or
retrieval quality on a held-out dataset.

For the inference timing output, disable pytest's output capture:

```bash
uv run pytest -q -s tests/test_tiny_model_inference.py tests/test_tiny_model_inference_8_devices.py
```

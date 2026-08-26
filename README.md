# A GPT-2-Style Language Model from Scratch

A small, educational language-model training pipeline built from first principles with **Python** and **PyTorch**. The goal is to understand the complete path from raw text to a trained decoder-only Transformer, rather than hiding the important pieces behind a high-level framework.

> **Status:** Early-stage / work in progress. The current notebook loads the training data; model, tokenizer, batching, training, and generation components are being built incrementally.

## Goals

This project is intended to implement a basic GPT-2-style pipeline, including:

- text dataset loading and preprocessing;
- tokenization and vocabulary handling;
- causal language-model training examples;
- masked self-attention;
- a decoder-only Transformer;
- optimization, evaluation, and checkpointing; and
- autoregressive text generation.

The implementation will favor readable code and experimentation over production-scale training performance.

## GPT-2-style architecture

The planned model is a decoder-only Transformer trained with next-token prediction:

```text
text → tokenizer → token IDs → batches of (x, y)
                              ↓
                    decoder-only Transformer
                              ↓
                    logits for the next token
                              ↓
                 cross-entropy language-model loss
```

Each target sequence is the input sequence shifted one position to the left. A causal attention mask prevents a token from looking at future tokens during training. The Transformer will use token and positional embeddings, pre-normalization, multi-head self-attention, an MLP block, residual connections, and a language-model head.

## Repository layout

```text
.
├── llm.ipynb          # Current exploratory data-loading notebook
├── src/llm/            # Python package (training code will live here)
├── pyproject.toml      # Project metadata and dependencies
├── uv.lock             # Reproducible dependency lockfile
└── README.md
```

## Dataset

The initial experiment uses the [`codelion/fineweb-edu-1B`](https://huggingface.co/datasets/codelion/fineweb-edu-1B) dataset through the Hugging Face `datasets` library:

```python
from datasets import load_dataset

dataset = load_dataset("codelion/fineweb-edu-1B", split="train")
```

The dataset contains a `text` column. It is large, so loading it may require significant disk space, memory, and download time. For local development, use a small subset before attempting a full training run.

## Requirements

- Python 3.13 or newer
- [uv](https://docs.astral.sh/uv/)
- A CPU or CUDA-capable PyTorch installation
- Enough storage and memory for the selected dataset

## Setup

Clone the repository and install its dependencies with `uv`:

```bash
git clone <repository-url>
cd llm
uv sync
```

Activate the environment if desired:

```bash
source .venv/bin/activate
```

## Explore the current notebook

Launch Jupyter using the project environment:

```bash
uv run jupyter lab
```

Open [`llm.ipynb`](./llm.ipynb) and run the cells to inspect the dataset and its text samples. The notebook is currently the main entry point for exploration.

## Planned training workflow

The pipeline will be developed in roughly these stages:

1. **Tokenizer** — convert text into stable token IDs and add encode/decode helpers.
2. **Dataset preparation** — concatenate or chunk documents into fixed-length sequences and create shifted targets.
3. **Model** — implement GPT-2-style embeddings, masked multi-head attention, feed-forward blocks, and output logits.
4. **Training loop** — add mini-batching, AdamW, learning-rate scheduling, gradient clipping, validation loss, and device selection.
5. **Checkpointing** — save model configuration, weights, optimizer state, and training progress.
6. **Generation** — sample tokens autoregressively with temperature and optional top-k filtering.
7. **Evaluation** — compare training and validation loss and run small qualitative generation tests.

## Example future training command

Once the training entry point is implemented, the intended interface will look similar to:

```bash
uv run python -m llm.train \
  --dataset codelion/fineweb-edu-1B \
  --block-size 256 \
  --batch-size 8 \
  --max-steps 10000
```

The command above is illustrative and is not available yet.

## Learning notes

This project is inspired by the original GPT-2 design, but it is not intended to reproduce OpenAI's GPT-2 checkpoints or training scale. A useful first experiment is a tiny model trained on a small data slice; once the end-to-end pipeline works, model size, context length, and dataset size can be increased gradually.

## License

No license has been selected yet.

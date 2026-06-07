# nanochat — Windows CPU Exploration

[![Lint](https://github.com/szavads/mywinnanochat/actions/workflows/lint.yml/badge.svg?branch=windows-cpu-support)](https://github.com/szavads/mywinnanochat/actions/workflows/lint.yml)

> **Fork of [karpathy/nanochat](https://github.com/karpathy/nanochat)**  
> Adapted for CPU-only environments on Windows. Includes bug diagnosis, fixes, and a full walkthrough of the LLM training pipeline.

---

## Overview

This fork documents my hands-on exploration of the full nanochat pipeline — pretraining, supervised fine-tuning (SFT), and chat inference — on a CPU-only Windows machine (Intel UHD 620, no CUDA).

The goal was to understand the end-to-end architecture of a modern LLM training stack, identify portability issues on constrained hardware, and contribute targeted fixes.

---

## What I Explored

### LLM Pipeline (end to end)

| Stage | Script | Purpose |
|---|---|---|
| Tokenizer training | `scripts/train_tokenizer.py` | BPE tokenizer on FineWeb-edu |
| Data preparation | `scripts/prepare_base_data.py` | Downloads climbmix dataset shards |
| Pretraining | `scripts/base_train.py` | Next-token prediction on 16K-token batches |
| SFT | `scripts/chat_sft.py` | Fine-tunes on dialogue data (SmolTalk + identity conversations) |
| Chat inference | `scripts/chat_cli.py` | Interactive chat with the trained model |

### Architecture Details

nanochat is not a vanilla Transformer. Key design choices I studied:

- **Sliding window attention** (`--window-pattern=L`) — local attention reduces compute for long sequences
- **RoPE (Rotary Position Embeddings)** — replaces learned absolute position embeddings
- **RMSNorm** — faster than LayerNorm, used before each attention and MLP block
- **ReLU²** activations — squared ReLU in MLP layers for sparsity
- **Value Embeddings** — adds ~600M parameters at near-zero FLOPs cost by injecting token embeddings into attention values
- **Muon optimizer** — Nesterov momentum + Polar Express orthogonalization for weight matrices; AdamW for embeddings and scalars
- **Split optimizer design** — different optimizers for different parameter types based on their geometry

### Gradient Accumulation

With `--total-batch-size=16384` and `--device-batch-size=4`:
- Each micro-step processes 4 × 512 = 2048 tokens
- Gradient accumulation steps = 16384 / 2048 = **8 steps per update**
- This keeps memory usage low while maintaining the effective batch size

---

## Key Findings & Fixes

### 1. `torch.compile` causes NaN loss on Windows CPU

**Symptom:** `loss: nan` from step 1 in `chat_sft.py`, even with `TORCHDYNAMO_DISABLE=1` set.

**Diagnosis:**
- Verified model weights were clean (no NaN/Inf in checkpoint)
- Tested forward pass directly: `loss: 12.89, nan: False` — model itself was correct
- Identified that `chat_sft.py` called `torch.compile(model, dynamic=False)` unconditionally, bypassing the env var guard

**Root cause:** `torch.compile` in `chat_sft.py` was called before the training loop regardless of `TORCHDYNAMO_DISABLE`. On Windows CPU without a C++ toolchain, this produces numerically incorrect compiled kernels rather than falling back gracefully.

**Fix** ([`scripts/chat_sft.py`](scripts/chat_sft.py)):
```python
# Before
model = torch.compile(model, dynamic=False)

# After
if not os.environ.get("TORCHDYNAMO_DISABLE"):
    model = torch.compile(model, dynamic=False)
```

### 2. Silent loss of training progress

**Symptom:** After ~50 minutes of pretraining, no checkpoint was found in `base_checkpoints/`.

**Root cause:** `base_train.py` saves only at `last_step` by default. If training crashes mid-run, all progress is lost.

**Fix:** Use `--save-every=50` to checkpoint every 50 steps:
```powershell
python -m scripts.base_train ... --save-every=50 --num-iterations=100
```

### 3. LR schedule edge case on small SFT datasets

**Symptom:** `lrm: -0.10` (negative learning rate multiplier) visible in logs at step 13.

**Root cause:** `identity_conversations.jsonl` is ~48KB — only ~13 optimization steps worth of data with `total-batch-size=4096`. The warmdown schedule overshoots 100% progress and returns a negative multiplier, which corrupts updates.

**Mitigation:** Use `--warmup-ratio=0.0 --warmdown-ratio=0.0` with small SFT datasets, or increase `--total-batch-size` to stretch the dataset over more steps.

---

## How to Run on Windows CPU

### Prerequisites

```powershell
# Allow PowerShell scripts
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Install uv (package manager)
pip install uv

# Install dependencies (CPU-only PyTorch)
cd C:\path\to\nanochat
uv sync --extra cpu
.\.venv\Scripts\activate
```

### Required environment variables (set at the start of every session)

```powershell
$env:TORCHDYNAMO_DISABLE = "1"   # prevents torch.compile C++ errors (omp.h missing)
$env:WANDB_MODE = "offline"       # skips wandb login prompts
```

### Step 1 — Train tokenizer and download data

```powershell
python -m scripts.train_tokenizer
python -m scripts.prepare_base_data
```

### Step 2 — Pretraining

```powershell
python -m scripts.base_train `
  --depth=6 --head-dim=64 --window-pattern=L `
  --max-seq-len=512 --device-batch-size=4 --total-batch-size=16384 `
  --eval-every=100 --eval-tokens=524288 `
  --core-metric-every=-1 --sample-every=100 `
  --num-iterations=100 --save-every=50 --run=demo
```

Expected: ~52 minutes, ~447 tok/sec on Intel UHD 620.

### Step 3 — SFT

```powershell
python -m scripts.chat_sft `
  --load-optimizer=0 `
  --device-batch-size=4 --total-batch-size=4096 `
  --matrix-lr=0.00003 --embedding-lr=0.00003 --unembedding-lr=0.00003 `
  --warmup-ratio=0.0 --warmdown-ratio=0.0 `
  --eval-every=100 --eval-tokens=131072 `
  --chatcore-every=-1 --mmlu-epochs=0 --gsm8k-epochs=0 `
  --num-iterations=13 --run=demo
```

### Step 4 — Chat

```powershell
# Base model (language model mode — completes text)
python -m scripts.chat_cli --source base

# After SFT (assistant mode)
python -m scripts.chat_cli
```

---

## Results

| Metric | Value |
|---|---|
| Hardware | Intel UHD 620 (CPU only, no CUDA) |
| OS | Windows 11 |
| Model size | depth=6, head-dim=64, ~7M params |
| Pretraining steps | 100 |
| Pretraining time | 51.61 minutes |
| Throughput | 447 tok/sec |
| val/bpb (pretraining) | 1.981 |

---

## Key Takeaways

- `torch.compile` requires a C++ toolchain on Windows and must be explicitly guarded — setting `TORCHDYNAMO_DISABLE=1` alone is not sufficient in all code paths
- Small SFT datasets expose edge cases in LR schedules designed for large-scale training
- Checkpoint saving strategy matters: default end-only saves risk losing hours of compute
- The nanochat architecture shows how much efficiency can be gained by combining orthogonal improvements: better optimizer (Muon), better attention (sliding window), better embeddings (Value Embeddings), better data (FineWeb-edu)

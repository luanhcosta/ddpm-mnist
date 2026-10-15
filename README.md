# DDPM MNIST

A Denoising Diffusion Probabilistic Model (DDPM) trained on MNIST from scratch, with a focus on **measuring and understanding CPU-GPU communication** during training and inference.

This project was built as the practical counterpart to a blog post on how data moves between CPU and GPU during deep learning workloads. Every benchmark in `profiling/communication.py` was designed to produce a concrete, measurable result rather than a rule of thumb.

**Hardware:** NVIDIA GeForce RTX 5050 Laptop GPU (8 GB, Blackwell sm_120) · CUDA 12.8

---

## Results

**Generated samples after 100 epochs:**

The model generates diverse, recognizable digits from pure Gaussian noise across all 10 classes with no mode collapse.

**CPU-GPU communication benchmarks (trained model, median latency where applicable):**

| Benchmark | Result |
|---|---|
| H2D bandwidth — pinned vs pageable (1 MB) | 9.03 vs 3.75 GB/s — **2.41×** |
| H2D bandwidth — pinned vs pageable (100 MB) | 9.76 vs 8.37 GB/s — **1.17×** |
| MNIST batch H2D latency — pinned vs pageable | 0.158 vs 0.260 ms — **1.65×** |
| Input-only loop (`next` + H2D) — pinned vs pageable | 3.226 vs 0.600 ms |
| AMP training step — data H2D share | **0.134 ms of 140.867 ms = 0.1%** |
| AMP training step — backward pass share | **87.023 ms = 61.8%** |
| DDPM-1000, observe every step vs never | 4.844s vs 4.740s — **1.02×** |
| DDIM-50 vs DDPM-1000 | 0.241s vs 4.740s — **19.7×** |

Key insight: pinned memory makes the isolated H2D copy faster, but it does not
automatically make every input loop faster. Pinning has its own host-side work,
and true copy/compute overlap requires a pipeline designed for it. In this
workload, H2D is only 0.1% of an AMP training step; backward computation is the
dominant cost.

---

## Architecture

**U-Net noise predictor (~9.7M parameters):**

```
Input: (B, 1, 28, 28) noisy image + timestep t
├── Sinusoidal time embedding → MLP → dim 256
├── Encoder
│   ├── Level 0: 28×28, 64 ch  — 2× ResBlock
│   ├── Level 1: 14×14, 128 ch — 2× ResBlock
│   └── Level 2: 7×7,  256 ch  — 2× ResBlock + SelfAttention
├── Bottleneck: 7×7, 256 ch — ResBlock + SelfAttention + ResBlock
└── Decoder (with skip connections)
    ├── Level 2 → 1: upsample + 2× ResBlock
    ├── Level 1 → 0: upsample + 2× ResBlock
    └── Final: GroupNorm → SiLU → Conv → (B, 1, 28, 28)
```

**Diffusion process:** linear schedule, β₁=1e-4 → β_T=0.02, T=1000 timesteps.

---

## Setup

```bash
git clone https://github.com/luanhcosta/ddpm-mnist
cd ddpm-mnist
python3 -m venv .venv
source .venv/bin/activate

# RTX 5050 (Blackwell sm_120) requires PyTorch nightly with CUDA 12.8+
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
pip install -r requirements.txt
```

> For older GPUs (Ampere, Hopper): `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126`

---

## Usage

**Train:**
```bash
python scripts/train.py --epochs 100

# With automatic blog asset export at end of training
python scripts/train.py --epochs 100 --assets-dir /path/to/blog/static/images
```

**Generate samples from a checkpoint:**
```bash
# DDPM (1000 steps)
python scripts/sample.py --checkpoint results/checkpoints/epoch_100.pt

# DDIM (50 steps, ~19x faster)
python scripts/sample.py --checkpoint results/checkpoints/epoch_100.pt --ddim --steps 50

# With denoising GIF
python scripts/sample.py --checkpoint results/checkpoints/epoch_100.pt --gif
```

**Run all CPU-GPU communication benchmarks:**
```bash
# With trained model (recommended)
python scripts/benchmark.py \
  --checkpoint results/checkpoints/epoch_100.pt \
  --assets-dir /path/to/blog/static/images

# Without checkpoint (timing benchmarks are still valid)
python scripts/benchmark.py
```

**Generate conceptual blog assets (no model required):**
```bash
python scripts/generate_assets.py --assets-dir /path/to/blog/static/images
```

---

## Project Structure

```
ddpm-mnist/
├── config/config.py          # Dataclass-based configuration
├── models/
│   ├── unet.py               # U-Net architecture with time conditioning
│   └── diffusion.py          # Forward process, loss, reverse mean
├── data/dataset.py           # MNIST DataLoader with pin_memory control
├── training/trainer.py       # Training loop: AMP, grad clip, profiler
├── sampling/
│   └── sampler.py            # DDPM/DDIM sampling + configurable host observation
├── profiling/
│   └── communication.py      # 6 CPU-GPU benchmarks + torch.profiler trace
├── utils/
│   ├── visualization.py      # Image grids, GIF export, all benchmark plots
│   └── metrics.py            # CSV logger, GPU memory tracker
├── scripts/
│   ├── train.py              # Training entry point
│   ├── sample.py             # Image generation from checkpoint
│   ├── benchmark.py          # Run all benchmarks
│   └── generate_assets.py    # Blog assets that need no trained model
└── results/
    ├── checkpoints/          # Model checkpoints (.pt)
    ├── samples/              # Per-epoch generated grids + GIFs
    ├── profiling/            # benchmarks.json + profiler traces
    ├── metrics/              # train_log.csv
    └── plots/                # All benchmark plots
```

---

## Training Details

| Parameter | Value |
|---|---|
| Dataset | MNIST (60,000 training images) |
| Epochs | 100 |
| Batch size | 128 |
| Optimizer | Adam, lr=2e-4 |
| Mixed precision | AMP (float16) |
| Gradient clipping | 1.0 |
| Noise schedule | Linear, T=1000 |
| Loss | MSE on predicted noise |

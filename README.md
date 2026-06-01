# DDPM MNIST

A Denoising Diffusion Probabilistic Model (DDPM) trained on MNIST from scratch, with a focus on **measuring and understanding CPU-GPU communication** during training and inference.

This project was built as the practical counterpart to a blog post on how data moves between CPU and GPU during deep learning workloads. Every benchmark in `profiling/communication.py` was designed to produce a concrete, measurable result rather than a rule of thumb.

**Hardware:** NVIDIA GeForce RTX 5050 Laptop GPU (8 GB, Blackwell sm_120) · CUDA 12.6

---

## Results

**Generated samples after 100 epochs:**

The model generates diverse, recognizable digits from pure Gaussian noise across all 10 classes with no mode collapse.

**CPU-GPU communication benchmarks (trained model):**

| Benchmark | Result |
|---|---|
| H2D bandwidth — pinned vs pageable (1 MB) | 17.3 vs 6.9 GB/s — **2.5× faster** |
| H2D bandwidth — pinned vs pageable (100 MB) | 11.8 vs 10.6 GB/s — **1.1× faster** |
| DataLoader throughput — pin_memory=True vs False | 14,662 vs 8,292 batches/s — **1.78×** |
| Training step — data H2D share | **0.08 ms of 213 ms total = 0.0%** |
| Training step — backward pass share | **140 ms = 65.8% of step time** |
| DDPM-1000 with 1000 CPU syncs vs pure GPU | 3.94s vs 3.78s — **1.04× overhead** |
| DDIM-50 vs DDPM-1000 | 0.20s vs 3.78s — **19.2× faster** |

Key insight: for a ~10M parameter model on 28×28 images, CPU-GPU data transfer is not the bottleneck. The backward pass owns 66% of step time. The benefit of pinned memory is meaningful only at small tensor sizes (< 10 MB); beyond that, PCIe bandwidth saturates both.

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
│   └── sampler.py            # ddpm_sample, ddpm_sample_naive_sync,
│                             # ddpm_sample_no_sync, ddim_sample
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

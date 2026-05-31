"""Visualization utilities for DDPM training and blog post asset generation.

Every public function accepts an optional `assets_dir` parameter.  When set,
the image is saved to both its default path AND to `assets_dir/<clean_name>`
so the caller can point it directly at the blog's static/images directory.
"""

import shutil
from pathlib import Path

import imageio
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ── Internal helpers ──────────────────────────────────────────────────────────

def _to_uint8(tensor: torch.Tensor) -> np.ndarray:
    """Convert a [-1, 1] tensor (B, C, H, W) to uint8 numpy (B, H, W, C)."""
    img = (tensor.clamp(-1, 1) + 1) / 2
    img = (img * 255).byte().cpu()
    if img.shape[1] == 1:
        img = img.repeat(1, 3, 1, 1)
    return img.permute(0, 2, 3, 1).numpy()


def _save_fig(fig, primary_path: Path, assets_dir: Path | None, assets_name: str) -> None:
    primary_path = Path(primary_path)
    primary_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(primary_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    if assets_dir:
        assets_dir = Path(assets_dir)
        assets_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(primary_path, assets_dir / assets_name)
        print(f"  assets → {assets_dir}/{assets_name}")


def _maybe_copy(src: Path, assets_dir: Path | None, assets_name: str) -> None:
    if assets_dir is None:
        return
    assets_dir = Path(assets_dir)
    assets_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, assets_dir / assets_name)
    print(f"  assets → {assets_dir}/{assets_name}")


# ── Image grids ───────────────────────────────────────────────────────────────

def save_image_grid(
    images: torch.Tensor,
    path: Path | str,
    nrow: int = 8,
    title: str | None = None,
    assets_dir: Path | None = None,
    assets_name: str = "samples.png",
) -> None:
    """Save a batch of [-1,1] images as a grid PNG."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    imgs = _to_uint8(images)
    n = imgs.shape[0]
    ncol = nrow
    nrows_grid = (n + ncol - 1) // ncol

    fig, axes = plt.subplots(nrows_grid, ncol, figsize=(ncol * 1.2, nrows_grid * 1.2))
    axes = np.array(axes).reshape(-1)
    for i, ax in enumerate(axes):
        if i < n:
            ax.imshow(imgs[i])
        ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout(pad=0.1)
    _save_fig(fig, path, assets_dir, assets_name)


def save_denoising_gif(
    frames: list[torch.Tensor],
    path: Path | str,
    n_samples: int = 8,
    fps: int = 10,
    assets_dir: Path | None = None,
    assets_name: str = "denoising.gif",
) -> None:
    """Save denoising trajectory (noise → image) as an animated GIF."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    gif_frames = []
    for frame_tensor in frames:
        imgs = _to_uint8(frame_tensor[:n_samples])
        row = np.concatenate(list(imgs), axis=1)
        gif_frames.append(row)

    imageio.mimsave(str(path), gif_frames, fps=fps, loop=0)
    _maybe_copy(path, assets_dir, assets_name)


# ── Noise schedule ────────────────────────────────────────────────────────────

def plot_noise_schedule(
    betas: torch.Tensor,
    path: Path | str,
    assets_dir: Path | None = None,
    assets_name: str = "noise_schedule.png",
) -> None:
    """Four-panel plot of the linear noise schedule over T timesteps.

    Panels:
        1. β_t — raw noise variance added at each step
        2. ᾱ_t — cumulative signal retention (product of all α_s up to t)
        3. Signal vs noise — √ᾱ_t and √(1-ᾱ_t) on the same axis;
           the crossing point (~t=600 for linear schedule) marks where the
           image becomes more noise than signal
        4. SNR — ᾱ_t / (1-ᾱ_t) on a log scale; shows how the signal-to-noise
           ratio collapses exponentially toward the end of the schedule
    """
    betas = betas.cpu().float()
    T = len(betas)
    ts = torch.arange(T)

    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, 0)
    sqrt_abar = alphas_cumprod.sqrt()
    sqrt_1m_abar = (1.0 - alphas_cumprod).sqrt()
    snr = alphas_cumprod / (1.0 - alphas_cumprod).clamp(min=1e-8)

    fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
    palette = ["#5b8db8", "#e07b54", "#2ecc71", "#9b59b6"]

    axes[0].plot(ts, betas, color=palette[0], lw=1.5)
    axes[0].set_title(r"$\beta_t$ (noise added per step)")
    axes[0].set_xlabel("Timestep t")

    axes[1].plot(ts, alphas_cumprod, color=palette[1], lw=1.5)
    axes[1].set_title(r"$\bar{\alpha}_t$ (cumulative signal)")
    axes[1].set_xlabel("Timestep t")

    axes[2].plot(ts, sqrt_abar, color=palette[0], lw=1.5, label=r"$\sqrt{\bar{\alpha}_t}$ (signal)")
    axes[2].plot(ts, sqrt_1m_abar, color=palette[2], lw=1.5, label=r"$\sqrt{1-\bar{\alpha}_t}$ (noise)")
    crossing = int((sqrt_abar - sqrt_1m_abar).abs().argmin())
    axes[2].axvline(crossing, color="gray", lw=1, ls="--", alpha=0.7)
    axes[2].text(crossing + 20, 0.55, f"t≈{crossing}", fontsize=8, color="gray")
    axes[2].set_title("Signal vs Noise amplitude")
    axes[2].set_xlabel("Timestep t")
    axes[2].legend(fontsize=8)

    axes[3].semilogy(ts, snr.clamp(min=1e-4), color=palette[3], lw=1.5)
    axes[3].set_title("SNR (log scale)")
    axes[3].set_xlabel("Timestep t")
    axes[3].yaxis.set_minor_formatter(ticker.NullFormatter())

    for ax in axes:
        ax.grid(alpha=0.3)
        ax.set_xlim(0, T)

    fig.suptitle("DDPM Linear Noise Schedule (T=1000)", fontsize=12, y=1.02)
    fig.tight_layout()
    _save_fig(fig, path, assets_dir, assets_name)


# ── Noising trajectory ────────────────────────────────────────────────────────

def plot_noising_trajectory(
    images_at_steps: list[tuple[int, torch.Tensor]],
    path: Path | str,
    assets_dir: Path | None = None,
    assets_name: str = "noising_trajectory.png",
) -> None:
    """Show one image being progressively noised across timesteps.

    Args:
        images_at_steps: list of (t, image_tensor) where image_tensor is (1, H, W)
                         in [-1, 1] range. Caller applies q_sample for each t.
    """
    n = len(images_at_steps)
    fig, axes = plt.subplots(1, n, figsize=(n * 2, 2.5))

    for ax, (t, img) in zip(axes, images_at_steps):
        arr = _to_uint8(img.unsqueeze(0))[0]
        ax.imshow(arr, cmap="gray")
        ax.set_title(f"t = {t}", fontsize=10)
        ax.axis("off")

    fig.suptitle("Forward Process: Adding Noise Step by Step", fontsize=11)
    fig.tight_layout()
    _save_fig(fig, path, assets_dir, assets_name)


# ── Training metrics ──────────────────────────────────────────────────────────

def plot_loss_curve(
    losses: list[float],
    path: Path | str,
    smoothing: int = 50,
    assets_dir: Path | None = None,
    assets_name: str = "loss_curve.png",
) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    steps = range(len(losses))
    ax.plot(steps, losses, alpha=0.25, color="#5b8db8", lw=0.7, label="raw")
    if smoothing > 1 and len(losses) > smoothing:
        kernel = np.ones(smoothing) / smoothing
        smoothed = np.convolve(losses, kernel, mode="valid")
        ax.plot(range(smoothing - 1, len(losses)), smoothed, color="#5b8db8", lw=2,
                label=f"smoothed (k={smoothing})")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("MSE Loss")
    ax.set_title("DDPM Training Loss (MNIST)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, path, assets_dir, assets_name)


# ── Benchmark plots ───────────────────────────────────────────────────────────

def plot_h2d_bandwidth(
    benchmark_results: dict,
    path: Path | str,
    assets_dir: Path | None = None,
    assets_name: str = "h2d_bandwidth.png",
) -> None:
    rows = benchmark_results["h2d_transfer"]
    labels = [f"{r['size_mb']} MB" for r in rows]
    pageable = [r["pageable_gbps"] for r in rows]
    pinned = [r["pinned_gbps"] for r in rows]

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 4))
    b1 = ax.bar(x - width / 2, pageable, width, label="Pageable memory", color="#5b8db8")
    b2 = ax.bar(x + width / 2, pinned, width, label="Pinned memory", color="#e07b54")
    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Bandwidth (GB/s)")
    ax.set_title("Host→Device Transfer Bandwidth: Pageable vs Pinned Memory")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, path, assets_dir, assets_name)


def plot_dataloader_comparison(
    benchmark_results: dict,
    path: Path | str,
    assets_dir: Path | None = None,
    assets_name: str = "dataloader_comparison.png",
) -> None:
    rows = benchmark_results["dataloader"]
    labels = ["pin_memory=False", "pin_memory=True"]
    times = [r["avg_ms_per_batch"] for r in rows]
    bps = [r["batches_per_second"] for r in rows]
    speedup = benchmark_results["pin_memory_speedup"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    colors = ["#c0392b", "#2ecc71"]

    bars = ax1.bar(labels, times, color=colors, edgecolor="white", width=0.5)
    for bar, val in zip(bars, times):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                 f"{val:.3f} ms", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Avg time per batch (ms)")
    ax1.set_title("Batch Transfer Time")
    ax1.grid(axis="y", alpha=0.3)

    bars = ax2.bar(labels, bps, color=colors, edgecolor="white", width=0.5)
    for bar, val in zip(bars, bps):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                 f"{val:.0f}/s", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Batches per second")
    ax2.set_title(f"Throughput  (speedup: {speedup:.2f}×)")
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("DataLoader: Effect of pin_memory on CPU→GPU Transfer", fontsize=11)
    fig.tight_layout()
    _save_fig(fig, path, assets_dir, assets_name)


def plot_sampling_comparison(
    benchmark_results: dict,
    path: Path | str,
    assets_dir: Path | None = None,
    assets_name: str = "sampling_comparison.png",
) -> None:
    s = benchmark_results["sampling_loop"]
    labels = ["DDPM-1000\n(1000 CPU syncs)", "DDPM-1000\n(pure GPU)", "DDIM-50\n(pure GPU)"]
    values = [s["ddpm_1000_naive_sync_s"], s["ddpm_1000_pure_gpu_s"], s["ddim_50_s"]]
    colors = ["#c0392b", "#2ecc71", "#3498db"]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(labels, values, color=colors, edgecolor="white", width=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.2f}s", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylabel("Time (seconds)")
    ax.set_title(
        f"Sampling {s['n_samples']} images — Sync overhead: {s['sync_overhead_factor']:.2f}× | "
        f"DDIM speedup: {s['ddim_speedup_vs_ddpm']:.1f}×"
    )
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, path, assets_dir, assets_name)


def plot_training_breakdown(
    benchmark_results: dict,
    path: Path | str,
    assets_dir: Path | None = None,
    assets_name: str = "training_breakdown.png",
) -> None:
    b = benchmark_results["training_step_breakdown"]
    phases = b["phases_avg_ms"]
    labels = list(phases.keys())
    values = list(phases.values())
    colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    bars = ax1.barh(labels, values, color=colors)
    for bar, val in zip(bars, values):
        pct = val / b["total_ms"] * 100
        ax1.text(val + 0.05, bar.get_y() + bar.get_height() / 2,
                 f"{val:.2f} ms  ({pct:.1f}%)", va="center", fontsize=9)
    ax1.set_xlabel("Time (ms)")
    ax1.set_title("Training Step — Phase Breakdown")
    ax1.grid(axis="x", alpha=0.3)
    ax1.set_xlim(right=max(values) * 1.5)

    ax2.pie(values, labels=labels, colors=colors, autopct="%1.1f%%",
            startangle=90, textprops={"fontsize": 9})
    ax2.set_title(f"Time Distribution\n(total: {b['total_ms']:.2f} ms/step)")

    note = f"H2D transfer = {b['data_h2d_pct']:.1f}% of total step time"
    fig.text(0.5, -0.03, note, ha="center", fontsize=10, style="italic", color="#555")
    fig.tight_layout()
    _save_fig(fig, path, assets_dir, assets_name)

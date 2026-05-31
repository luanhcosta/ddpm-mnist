from pathlib import Path

import imageio
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def _to_uint8(tensor: torch.Tensor) -> np.ndarray:
    """Convert a [-1, 1] tensor (B, C, H, W) to uint8 numpy (B, H, W, C)."""
    img = (tensor.clamp(-1, 1) + 1) / 2  # → [0, 1]
    img = (img * 255).byte().cpu()
    if img.shape[1] == 1:
        img = img.repeat(1, 3, 1, 1)  # grayscale → RGB for visualization
    return img.permute(0, 2, 3, 1).numpy()


def save_image_grid(
    images: torch.Tensor,
    path: Path | str,
    nrow: int = 8,
    title: str | None = None,
) -> None:
    """Save a batch of images as a grid PNG."""
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
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_denoising_gif(
    frames: list[torch.Tensor],
    path: Path | str,
    n_samples: int = 8,
    fps: int = 10,
) -> None:
    """Save the denoising trajectory (noise → image) as an animated GIF.

    Args:
        frames: list of (B, C, H, W) tensors from ddpm_sample(capture_every=N)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    gif_frames = []
    for frame_tensor in frames:
        imgs = _to_uint8(frame_tensor[:n_samples])
        # Tile horizontally
        row = np.concatenate(list(imgs), axis=1)
        gif_frames.append(row)

    imageio.mimsave(str(path), gif_frames, fps=fps, loop=0)


def plot_loss_curve(
    losses: list[float],
    path: Path | str,
    smoothing: int = 50,
) -> None:
    """Plot training loss over steps with optional smoothing."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    steps = range(len(losses))
    ax.plot(steps, losses, alpha=0.3, color="steelblue", lw=0.8, label="raw")

    if smoothing > 1 and len(losses) > smoothing:
        kernel = np.ones(smoothing) / smoothing
        smoothed = np.convolve(losses, kernel, mode="valid")
        ax.plot(range(smoothing - 1, len(losses)), smoothed, color="steelblue", lw=2, label=f"smoothed (k={smoothing})")

    ax.set_xlabel("Step")
    ax.set_ylabel("MSE Loss")
    ax.set_title("DDPM Training Loss")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_benchmark_bar(
    labels: list[str],
    values: list[float],
    path: Path | str,
    title: str,
    ylabel: str,
    color: str = "steelblue",
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.5), 4))
    bars = ax.bar(labels, values, color=color, edgecolor="white", width=0.6)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_h2d_bandwidth(benchmark_results: dict, path: Path | str) -> None:
    """Generate bandwidth comparison chart from h2d_transfer results."""
    rows = benchmark_results["h2d_transfer"]
    labels = [f"{r['size_mb']}MB" for r in rows]
    pageable = [r["pageable_gbps"] for r in rows]
    pinned = [r["pinned_gbps"] for r in rows]

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x - width / 2, pageable, width, label="Pageable", color="#5b8db8")
    ax.bar(x + width / 2, pinned, width, label="Pinned", color="#e07b54")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Bandwidth (GB/s)")
    ax.set_title("H2D Transfer Bandwidth: Pageable vs Pinned Memory")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_sampling_comparison(benchmark_results: dict, path: Path | str) -> None:
    """Bar chart of sampling strategies by wall-clock time."""
    s = benchmark_results["sampling_loop"]
    labels = ["DDPM-1000\n(naive sync)", "DDPM-1000\n(pure GPU)", "DDIM-50\n(pure GPU)"]
    values = [s["ddpm_1000_naive_sync_s"], s["ddpm_1000_pure_gpu_s"], s["ddim_50_s"]]
    colors = ["#c0392b", "#2ecc71", "#3498db"]

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, values, color=colors, edgecolor="white", width=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                f"{val:.2f}s", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("Time (seconds)")
    ax.set_title(f"Sampling Time for {s['n_samples']} Images")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_training_breakdown(benchmark_results: dict, path: Path | str) -> None:
    """Pie + bar chart of training step phase breakdown."""
    b = benchmark_results["training_step_breakdown"]
    phases = b["phases_avg_ms"]

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    labels = list(phases.keys())
    values = list(phases.values())
    colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Bar chart
    ax1.barh(labels, values, color=colors)
    for i, (label, val) in enumerate(zip(labels, values)):
        pct = val / b["total_ms"] * 100
        ax1.text(val + 0.05, i, f"{val:.2f}ms ({pct:.1f}%)", va="center", fontsize=9)
    ax1.set_xlabel("Time (ms)")
    ax1.set_title("Training Step Phase Breakdown")
    ax1.grid(axis="x", alpha=0.3)

    # Pie chart
    ax2.pie(values, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
    ax2.set_title("Time Distribution")

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)

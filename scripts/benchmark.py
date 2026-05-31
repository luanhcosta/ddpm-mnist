"""Run all CPU-GPU communication benchmarks.

This script generates all the data needed for the blog post section on
CPU-GPU communication during DDPM training and inference.

Usage:
    # Run all benchmarks (requires a trained checkpoint for sampling benchmarks)
    python scripts/benchmark.py --checkpoint results/checkpoints/epoch_100.pt

    # Skip sampling benchmark (runs even without a checkpoint)
    python scripts/benchmark.py --no-sampling

    # Generate plots from existing results
    python scripts/benchmark.py --plots-only results/profiling/benchmarks.json
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import Config, ModelConfig, DiffusionConfig
from data.dataset import get_dataloader
from models.unet import UNet
from models.diffusion import DiffusionProcess
from profiling.communication import run_all_benchmarks
from utils.visualization import (
    plot_h2d_bandwidth,
    plot_dataloader_comparison,
    plot_sampling_comparison,
    plot_training_breakdown,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None, help="Path to model checkpoint for sampling benchmarks")
    p.add_argument("--no-sampling", action="store_true", help="Skip sampling loop benchmark")
    p.add_argument("--plots-only", default=None, help="Path to existing benchmarks.json, skip running")
    p.add_argument("--output-dir", default="results/profiling")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--assets-dir", default=None,
        help="Optional path to copy blog-ready images to (e.g. ../blog-luan-costa/static/images/ddpm)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    assets_dir = Path(args.assets_dir) if args.assets_dir else None

    if args.plots_only:
        with open(args.plots_only) as f:
            results = json.load(f)
        _generate_plots(results, output_dir, assets_dir)
        return

    print(f"Running benchmarks on {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")

    cfg = Config()
    cfg_model = ModelConfig()
    cfg_diff = DiffusionConfig()

    model = UNet(cfg_model).to(device)

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        print(f"Loaded checkpoint from epoch {ckpt['epoch']}")
    else:
        print("No checkpoint provided — using random weights (still valid for timing benchmarks)")

    model.eval()
    diffusion = DiffusionProcess(cfg_diff, device)

    results = run_all_benchmarks(cfg, model, diffusion, device, output_dir)
    _generate_plots(results, output_dir, assets_dir)


def _generate_plots(results: dict, output_dir: Path, assets_dir: Path | None = None) -> None:
    print("\nGenerating plots...")
    plots_dir = output_dir.parent / "plots"

    if "h2d_transfer" in results:
        plot_h2d_bandwidth(results, plots_dir / "h2d_bandwidth.png",
                           assets_dir=assets_dir, assets_name="h2d_bandwidth.png")

    if "dataloader" in results:
        plot_dataloader_comparison(results, plots_dir / "dataloader_comparison.png",
                                   assets_dir=assets_dir, assets_name="dataloader_comparison.png")

    if "sampling_loop" in results:
        plot_sampling_comparison(results, plots_dir / "sampling_comparison.png",
                                 assets_dir=assets_dir, assets_name="sampling_comparison.png")

    if "training_step_breakdown" in results:
        plot_training_breakdown(results, plots_dir / "training_breakdown.png",
                                assets_dir=assets_dir, assets_name="training_breakdown.png")


if __name__ == "__main__":
    main()

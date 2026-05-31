"""Generate images from a saved checkpoint.

Usage:
    python scripts/sample.py --checkpoint results/checkpoints/epoch_100.pt
    python scripts/sample.py --checkpoint results/checkpoints/epoch_100.pt --ddim --steps 50
    python scripts/sample.py --checkpoint results/checkpoints/epoch_100.pt --gif
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import Config, ModelConfig, DiffusionConfig
from models.unet import UNet
from models.diffusion import DiffusionProcess
from sampling.sampler import ddpm_sample, ddim_sample
from utils.visualization import save_image_grid, save_denoising_gif


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--n-samples", type=int, default=64)
    p.add_argument("--ddim", action="store_true", help="Use DDIM instead of DDPM")
    p.add_argument("--steps", type=int, default=50, help="DDIM steps")
    p.add_argument("--gif", action="store_true", help="Save denoising GIF (DDPM only)")
    p.add_argument("--output", default="results/samples/manual")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}")

    cfg_model = ModelConfig()
    cfg_diff = DiffusionConfig()

    model = UNet(cfg_model).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    diffusion = DiffusionProcess(cfg_diff, device)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.ddim:
        print(f"Sampling with DDIM ({args.steps} steps)...")
        images, elapsed = ddim_sample(model, diffusion, args.n_samples, device, n_steps=args.steps)
        print(f"Done in {elapsed:.2f}s")
        save_image_grid(images, out_dir / f"ddim_{args.steps}steps.png", title=f"DDIM-{args.steps}")
    else:
        print("Sampling with DDPM (1000 steps)...")
        capture = 100 if args.gif else 0
        images, frames = ddpm_sample(model, diffusion, args.n_samples, device, capture_every=capture)
        save_image_grid(images, out_dir / "ddpm_samples.png", title="DDPM-1000")
        if args.gif and frames:
            save_denoising_gif(frames, out_dir / "denoising.gif")
            print(f"GIF saved to {out_dir}/denoising.gif")

    print(f"Images saved to {out_dir}")


if __name__ == "__main__":
    main()

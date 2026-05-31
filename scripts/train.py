"""Entry point for DDPM training.

Usage:
    python scripts/train.py
    python scripts/train.py --epochs 50 --batch-size 256
    python scripts/train.py --profile-steps 10    # enable torch.profiler
"""

import argparse
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import Config, DataConfig, ModelConfig, DiffusionConfig, TrainConfig
from data.dataset import get_dataloader
from models.unet import UNet
from models.diffusion import DiffusionProcess
from training.trainer import Trainer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--base-channels", type=int, default=64)
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--schedule", choices=["linear", "cosine"], default="linear")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--sample-every", type=int, default=5)
    p.add_argument("--profile-steps", type=int, default=0)
    p.add_argument("--no-pin-memory", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    cfg = Config(
        data=DataConfig(batch_size=args.batch_size, pin_memory=not args.no_pin_memory),
        model=ModelConfig(base_channels=args.base_channels),
        diffusion=DiffusionConfig(timesteps=args.timesteps, schedule=args.schedule),
        train=TrainConfig(
            epochs=args.epochs,
            learning_rate=args.lr,
            save_every=args.save_every,
            sample_every=args.sample_every,
            profile_steps=args.profile_steps,
        ),
        device=args.device,
        seed=args.seed,
    )

    device = torch.device(cfg.device)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
        print(f"VRAM: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")

    model = UNet(cfg.model).to(device)
    diffusion = DiffusionProcess(cfg.diffusion, device)
    dataloader = get_dataloader(cfg.data)

    trainer = Trainer(cfg, model, diffusion, dataloader)
    trainer.train()


if __name__ == "__main__":
    main()

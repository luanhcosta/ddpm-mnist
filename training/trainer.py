import time
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from config.config import Config
from models.unet import UNet
from models.diffusion import DiffusionProcess
from sampling.sampler import ddpm_sample
from utils.visualization import save_image_grid, save_denoising_gif, plot_loss_curve
from utils.metrics import MetricsLogger, GPUMemoryTracker


class Trainer:
    def __init__(
        self,
        cfg: Config,
        model: UNet,
        diffusion: DiffusionProcess,
        dataloader: torch.utils.data.DataLoader,
        assets_dir: Path | None = None,
    ):
        self.cfg = cfg
        self.model = model
        self.diffusion = diffusion
        self.dataloader = dataloader
        self.device = torch.device(cfg.device)

        self.optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.learning_rate)
        self.scaler = torch.cuda.amp.GradScaler()

        ckpt_dir = Path(cfg.train.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        results_dir = Path(cfg.train.results_dir)
        (results_dir / "samples").mkdir(parents=True, exist_ok=True)
        (results_dir / "plots").mkdir(parents=True, exist_ok=True)

        self.ckpt_dir = ckpt_dir
        self.results_dir = results_dir
        self.assets_dir = Path(assets_dir) if assets_dir else None
        self.logger = MetricsLogger(results_dir / "metrics" / "train_log.csv")
        self.mem = GPUMemoryTracker(cfg.device)

        self.global_step = 0
        self.all_losses: list[float] = []

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        epoch_loss = 0.0
        n_batches = 0

        profiler_ctx = self._make_profiler(epoch) if self.cfg.train.profile_steps > 0 else None
        if profiler_ctx is not None:
            profiler_ctx.__enter__()

        pbar = tqdm(self.dataloader, desc=f"Epoch {epoch:03d}", leave=False)
        for x, _ in pbar:
            # ── CPU→GPU transfer ─────────────────────────────────────────────
            # non_blocking=True overlaps transfer with other GPU work in the
            # default stream. Requires pin_memory=True on the DataLoader side.
            x = x.to(self.device, non_blocking=self.cfg.data.pin_memory)
            t = torch.randint(0, self.diffusion.T, (x.size(0),), device=self.device)

            # ── Forward + loss ───────────────────────────────────────────────
            self.optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                loss = self.diffusion.loss(self.model, x, t)

            # ── Backward + clip ──────────────────────────────────────────────
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.gradient_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            loss_val = loss.item()  # sync point — pulls scalar from GPU
            epoch_loss += loss_val
            self.all_losses.append(loss_val)
            self.global_step += 1
            n_batches += 1

            pbar.set_postfix(loss=f"{loss_val:.4f}")
            self.logger.log(
                epoch=epoch,
                step=self.global_step,
                loss=round(loss_val, 6),
                **self.mem.snapshot(),
            )

            if profiler_ctx is not None and self.global_step <= self.cfg.train.profile_steps + 2:
                profiler_ctx.step()

        if profiler_ctx is not None:
            profiler_ctx.__exit__(None, None, None)

        return epoch_loss / max(n_batches, 1)

    def train(self) -> None:
        print(f"Model parameters: {self.model.count_parameters():,}")
        print(f"Training on: {self.device}")
        self.mem.reset_peak()

        for epoch in range(1, self.cfg.train.epochs + 1):
            t0 = time.perf_counter()
            avg_loss = self.train_epoch(epoch)
            epoch_time = time.perf_counter() - t0

            mem = self.mem.snapshot()
            print(
                f"Epoch {epoch:03d}/{self.cfg.train.epochs}  "
                f"loss={avg_loss:.4f}  "
                f"time={epoch_time:.1f}s  "
                f"GPU={mem['allocated_mb']}MB"
            )

            if epoch % self.cfg.train.save_every == 0 or epoch == self.cfg.train.epochs:
                self.save_checkpoint(epoch)

            if epoch % self.cfg.train.sample_every == 0 or epoch == self.cfg.train.epochs:
                self._generate_samples(epoch)

        # Final artifacts
        plot_loss_curve(
            self.all_losses,
            self.results_dir / "plots" / "loss_curve.png",
            assets_dir=self.assets_dir,
            assets_name="loss_curve.png",
        )
        self.logger.close()
        print("Training complete.")

    def save_checkpoint(self, epoch: int) -> None:
        path = self.ckpt_dir / f"epoch_{epoch:03d}.pt"
        torch.save({
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "loss_history": self.all_losses,
        }, path)

    @torch.no_grad()
    def _generate_samples(self, epoch: int) -> None:
        self.model.eval()
        n = self.cfg.train.n_samples
        samples_dir = self.results_dir / "samples" / f"epoch_{epoch:03d}"
        samples_dir.mkdir(parents=True, exist_ok=True)

        # Sample grid
        images, frames = ddpm_sample(
            self.model, self.diffusion, n, self.device,
            capture_every=100,
        )
        is_final = epoch == self.cfg.train.epochs
        save_image_grid(
            images, samples_dir / "grid.png", title=f"Epoch {epoch}",
            assets_dir=self.assets_dir if is_final else None,
            assets_name="samples_final.png",
        )

        # Denoising GIF (only a few samples)
        if frames:
            save_denoising_gif(
                frames, samples_dir / "denoising.gif", n_samples=8,
                assets_dir=self.assets_dir if is_final else None,
                assets_name="denoising.gif",
            )

        self.model.train()

    def _make_profiler(self, epoch: int):
        output_dir = self.results_dir / "profiling" / f"epoch_{epoch:03d}"
        output_dir.mkdir(parents=True, exist_ok=True)
        return torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=self.cfg.train.profile_steps),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(str(output_dir)),
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        )

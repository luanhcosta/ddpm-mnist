import time
import torch

from models.unet import UNet
from models.diffusion import DiffusionProcess


@torch.no_grad()
def ddpm_sample(
    model: UNet,
    diffusion: DiffusionProcess,
    n_samples: int,
    device: torch.device,
    capture_every: int = 0,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Full 1000-step DDPM reverse process.

    Args:
        capture_every: if > 0, stores intermediate x_t every N steps for
                       visualization of the denoising trajectory.

    Returns:
        (final_images, frames) where frames is a list of tensors for GIF export.
    """
    model.eval()
    shape = (n_samples, diffusion.alphas.shape[0] and 1, 28, 28)
    shape = (n_samples, 1, 28, 28)
    x = torch.randn(shape, device=device)
    frames: list[torch.Tensor] = []

    for t_idx in reversed(range(diffusion.T)):
        t_batch = torch.full((n_samples,), t_idx, device=device, dtype=torch.long)

        predicted_noise = model(x, t_batch)
        mean = diffusion.p_mean(predicted_noise, x, t_batch)

        if t_idx > 0:
            noise = torch.randn_like(x)
            variance = diffusion.posterior_variance[t_idx]
            x = mean + torch.sqrt(variance) * noise
        else:
            x = mean

        if capture_every > 0 and t_idx % capture_every == 0:
            frames.append(x.cpu().clone())

    return x, frames


@torch.no_grad()
def ddpm_sample_naive_sync(
    model: UNet,
    diffusion: DiffusionProcess,
    n_samples: int,
    device: torch.device,
) -> tuple[torch.Tensor, float]:
    """DDPM sampling with a forced CPU synchronization at every step.

    Calling .item() on a GPU tensor forces cudaDeviceSynchronize() — the CPU
    blocks until the GPU finishes that operation before continuing.
    With T=1000 steps, this produces 1000 sync points.

    Returns (images, time_seconds).
    """
    model.eval()
    shape = (n_samples, 1, 28, 28)
    x = torch.randn(shape, device=device)

    torch.cuda.synchronize()
    t_start = time.perf_counter()

    for t_idx in reversed(range(diffusion.T)):
        t_batch = torch.full((n_samples,), t_idx, device=device, dtype=torch.long)
        predicted_noise = model(x, t_batch)
        mean = diffusion.p_mean(predicted_noise, x, t_batch)

        if t_idx > 0:
            noise = torch.randn_like(x)
            variance = diffusion.posterior_variance[t_idx]
            x = mean + torch.sqrt(variance) * noise
        else:
            x = mean

        # Force CPU-GPU sync: CPU reads a scalar from GPU memory at every step
        _ = x.mean().item()

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t_start
    return x, elapsed


@torch.no_grad()
def ddpm_sample_no_sync(
    model: UNet,
    diffusion: DiffusionProcess,
    n_samples: int,
    device: torch.device,
) -> tuple[torch.Tensor, float]:
    """DDPM sampling with no CPU synchronization during the loop.

    The GPU executes all 1000 denoising steps as a continuous stream of
    kernel launches. The CPU only synchronizes once at the end, when we
    call torch.cuda.synchronize() to get the accurate wall-clock time.

    Returns (images, time_seconds).
    """
    model.eval()
    shape = (n_samples, 1, 28, 28)
    x = torch.randn(shape, device=device)

    torch.cuda.synchronize()
    t_start = time.perf_counter()

    for t_idx in reversed(range(diffusion.T)):
        t_batch = torch.full((n_samples,), t_idx, device=device, dtype=torch.long)
        predicted_noise = model(x, t_batch)
        mean = diffusion.p_mean(predicted_noise, x, t_batch)

        if t_idx > 0:
            noise = torch.randn_like(x)
            variance = diffusion.posterior_variance[t_idx]
            x = mean + torch.sqrt(variance) * noise
        else:
            x = mean

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t_start
    return x, elapsed


@torch.no_grad()
def ddim_sample(
    model: UNet,
    diffusion: DiffusionProcess,
    n_samples: int,
    device: torch.device,
    n_steps: int = 50,
    eta: float = 0.0,
) -> tuple[torch.Tensor, float]:
    """DDIM deterministic sampling with a reduced step count.

    DDIM (Song et al., 2020) reformulates the reverse process so that a
    subset of T timesteps is sufficient — typically 50 steps vs 1000.
    η=0 makes sampling fully deterministic (no added noise).

    Returns (images, time_seconds).
    """
    model.eval()
    T = diffusion.T
    step_size = T // n_steps
    timesteps = list(reversed(range(0, T, step_size)))

    x = torch.randn((n_samples, 1, 28, 28), device=device)

    torch.cuda.synchronize()
    t_start = time.perf_counter()

    for i, t in enumerate(timesteps):
        t_batch = torch.full((n_samples,), t, device=device, dtype=torch.long)
        predicted_noise = model(x, t_batch)

        alpha_bar_t = diffusion.alphas_cumprod[t]
        alpha_bar_prev = (
            diffusion.alphas_cumprod[timesteps[i + 1]]
            if i + 1 < len(timesteps)
            else torch.tensor(1.0, device=device)
        )

        # Predict x0 from xt and the predicted noise
        pred_x0 = (x - (1 - alpha_bar_t).sqrt() * predicted_noise) / alpha_bar_t.sqrt()
        pred_x0 = pred_x0.clamp(-1.0, 1.0)

        # Variance of the next step
        sigma = eta * ((1 - alpha_bar_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_prev)).sqrt()
        dir_xt = (1 - alpha_bar_prev - sigma**2).sqrt() * predicted_noise

        noise = torch.randn_like(x) if eta > 0 else 0.0
        x = alpha_bar_prev.sqrt() * pred_x0 + dir_xt + sigma * noise

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t_start
    return x, elapsed

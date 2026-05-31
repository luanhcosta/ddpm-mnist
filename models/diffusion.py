import math
import torch
import torch.nn.functional as F

from config.config import DiffusionConfig


def make_beta_schedule(T: int, beta_start: float, beta_end: float, schedule: str) -> torch.Tensor:
    if schedule == "linear":
        return torch.linspace(beta_start, beta_end, T)
    if schedule == "cosine":
        steps = T + 1
        x = torch.linspace(0, T, steps)
        alphas_cumprod = torch.cos(((x / T) + 0.008) / 1.008 * math.pi / 2) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1.0 - alphas_cumprod[1:] / alphas_cumprod[:-1]
        return betas.clamp(1e-4, 0.9999)
    raise ValueError(f"Unknown schedule: {schedule}")


class DiffusionProcess:
    """Encapsulates the entire DDPM forward/reverse math.

    All schedule tensors are precomputed once on CPU and transferred to the
    target device at construction time. This is one of the most visible
    CPU→GPU transfers in the codebase — benchmarked in profiling/communication.py.

    Forward process (noising):
        q(x_t | x_0) = N(sqrt(ᾱ_t) * x_0, (1 - ᾱ_t) * I)
        Closed-form: x_t = sqrt(ᾱ_t) * x_0 + sqrt(1 - ᾱ_t) * ε

    Reverse process (denoising):
        p_θ(x_{t-1} | x_t) = N(μ_θ(x_t, t), σ_t² * I)
        μ_θ = (x_t - β_t / sqrt(1 - ᾱ_t) * ε_θ(x_t, t)) / sqrt(α_t)
        σ_t = sqrt(β_t) (simplified variance)
    """

    def __init__(self, cfg: DiffusionConfig, device: torch.device):
        T = cfg.timesteps

        # All computed on CPU, transferred once — measure this in benchmarks
        betas = make_beta_schedule(T, cfg.beta_start, cfg.beta_end, cfg.schedule)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.T = T
        self.device = device

        def reg(x: torch.Tensor) -> torch.Tensor:
            return x.float().to(device)

        self.betas = reg(betas)
        self.alphas = reg(alphas)
        self.alphas_cumprod = reg(alphas_cumprod)
        self.alphas_cumprod_prev = reg(alphas_cumprod_prev)

        # Precomputed terms used in forward and reverse passes
        self.sqrt_alphas_cumprod = reg(torch.sqrt(alphas_cumprod))
        self.sqrt_one_minus_alphas_cumprod = reg(torch.sqrt(1.0 - alphas_cumprod))
        self.sqrt_recip_alphas = reg(torch.sqrt(1.0 / alphas))

        # Posterior variance: β̃_t = β_t * (1 - ᾱ_{t-1}) / (1 - ᾱ_t)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.posterior_variance = reg(posterior_variance)
        self.posterior_log_variance = reg(torch.log(posterior_variance.clamp(min=1e-20)))

    def _extract(self, values: torch.Tensor, t: torch.Tensor, shape: tuple) -> torch.Tensor:
        """Pick schedule values at timesteps t and reshape for broadcasting."""
        out = values[t]
        while len(out.shape) < len(shape):
            out = out[..., None]
        return out

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        """Forward process: add noise to x0 at timestep t.

        x_t = sqrt(ᾱ_t) * x_0 + sqrt(1 - ᾱ_t) * ε
        """
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_abar = self._extract(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_1m_abar = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        return sqrt_abar * x0 + sqrt_1m_abar * noise

    def loss(self, model: torch.nn.Module, x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Simple MSE loss: predict the noise added at timestep t."""
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)
        predicted_noise = model(xt, t)
        return F.mse_loss(predicted_noise, noise)

    def p_mean(self, model_output: torch.Tensor, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Compute the denoising mean μ_θ(x_t, t)."""
        betas = self._extract(self.betas, t, xt.shape)
        sqrt_1m_abar = self._extract(self.sqrt_one_minus_alphas_cumprod, t, xt.shape)
        sqrt_recip_a = self._extract(self.sqrt_recip_alphas, t, xt.shape)
        return sqrt_recip_a * (xt - betas / sqrt_1m_abar * model_output)

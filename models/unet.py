import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config.config import ModelConfig


class SinusoidalEmbedding(nn.Module):
    """Encodes a scalar timestep t into a sinusoidal vector of `dim` dimensions.

    Same positional encoding used in the original Transformer, applied here to
    the diffusion timestep so the network can distinguish between noise levels.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / (half - 1)
        )
        emb = t.float()[:, None] * freqs[None]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class ResBlock(nn.Module):
    """Residual block with time-step conditioning.

    The time embedding is projected to `out_ch` and added after the first
    conv, injecting the current noise level into every feature map.
    """

    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int, groups: int = 8):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_emb_dim, out_ch)
        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.act = nn.SiLU()
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        h = h + self.time_proj(self.act(t_emb))[:, :, None, None]
        h = self.conv2(self.act(self.norm2(h)))
        return h + self.skip(x)


class SelfAttention(nn.Module):
    """Single-head self-attention over spatial positions.

    Applied only at the bottleneck (7×7) to capture global dependencies
    without the cost of full spatial attention at higher resolutions.
    """

    def __init__(self, channels: int, groups: int = 8):
        super().__init__()
        self.norm = nn.GroupNorm(groups, channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.proj = nn.Conv1d(channels, channels, 1)
        self.scale = channels**-0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x).reshape(B, C, -1)
        qkv = self.qkv(h).reshape(B, 3, C, H * W)
        q, k, v = qkv.unbind(1)
        attn = torch.softmax(torch.bmm(q.transpose(1, 2), k) * self.scale, dim=-1)
        out = torch.bmm(attn, v.transpose(1, 2)).transpose(1, 2)
        return x + self.proj(out).reshape(B, C, H, W)


class UNet(nn.Module):
    """U-Net noise predictor for 28×28 MNIST images.

    Architecture (C = base_channels = 64):
        Encoder:  28×28×1  → 28×28×C → 14×14×2C → 7×7×4C
        Bottleneck:         7×7×4C (ResBlock + Attention + ResBlock)
        Decoder:  7×7×4C  → 14×14×2C → 28×28×C → 28×28×1
    Skip connections concatenate encoder features into the decoder.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        C = cfg.base_channels
        te = cfg.time_emb_dim

        # Time embedding: scalar t → vector of dim `te`
        self.time_mlp = nn.Sequential(
            SinusoidalEmbedding(C),
            nn.Linear(C, te),
            nn.SiLU(),
            nn.Linear(te, te),
        )

        # ── Encoder ──────────────────────────────────────────────────────────
        self.init_conv = nn.Conv2d(cfg.channels, C, 3, padding=1)

        # Level 0: 28×28, C channels
        self.enc1 = nn.ModuleList([ResBlock(C, C, te), ResBlock(C, C, te)])
        self.down1 = nn.Conv2d(C, C, 3, stride=2, padding=1)

        # Level 1: 14×14, 2C channels
        self.enc2 = nn.ModuleList([ResBlock(C, C * 2, te), ResBlock(C * 2, C * 2, te)])
        self.down2 = nn.Conv2d(C * 2, C * 2, 3, stride=2, padding=1)

        # Level 2: 7×7, 4C channels (with attention)
        self.enc3 = nn.ModuleList([ResBlock(C * 2, C * 4, te), ResBlock(C * 4, C * 4, te)])
        self.enc3_attn = SelfAttention(C * 4)

        # ── Bottleneck ───────────────────────────────────────────────────────
        self.mid1 = ResBlock(C * 4, C * 4, te)
        self.mid_attn = SelfAttention(C * 4)
        self.mid2 = ResBlock(C * 4, C * 4, te)

        # ── Decoder ──────────────────────────────────────────────────────────
        # Level 2 → 1: upsample 7×7→14×14, cat with enc2 skip (2C), → 2C
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(C * 4, C * 4, 3, padding=1),
        )
        self.dec3 = nn.ModuleList(
            [ResBlock(C * 4 + C * 2, C * 4, te), ResBlock(C * 4, C * 2, te)]
        )

        # Level 1 → 0: upsample 14×14→28×28, cat with enc1 skip (C), → C
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(C * 2, C * 2, 3, padding=1),
        )
        self.dec2 = nn.ModuleList(
            [ResBlock(C * 2 + C, C * 2, te), ResBlock(C * 2, C, te)]
        )

        self.final_norm = nn.GroupNorm(8, C)
        self.final_conv = nn.Conv2d(C, cfg.channels, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(t)

        x = self.init_conv(x)

        # Encoder
        for block in self.enc1:
            x = block(x, t_emb)
        skip1 = x
        x = self.down1(x)

        for block in self.enc2:
            x = block(x, t_emb)
        skip2 = x
        x = self.down2(x)

        for block in self.enc3:
            x = block(x, t_emb)
        x = self.enc3_attn(x)

        # Bottleneck
        x = self.mid1(x, t_emb)
        x = self.mid_attn(x)
        x = self.mid2(x, t_emb)

        # Decoder
        x = self.up3(x)
        x = torch.cat([x, skip2], dim=1)
        for block in self.dec3:
            x = block(x, t_emb)

        x = self.up2(x)
        x = torch.cat([x, skip1], dim=1)
        for block in self.dec2:
            x = block(x, t_emb)

        return self.final_conv(F.silu(self.final_norm(x)))

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

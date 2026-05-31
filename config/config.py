from dataclasses import dataclass, field


@dataclass
class DataConfig:
    dataset: str = "mnist"
    data_dir: str = "./data_cache"
    image_size: int = 28
    channels: int = 1
    batch_size: int = 128
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class ModelConfig:
    channels: int = 1
    base_channels: int = 64
    time_emb_dim: int = 256


@dataclass
class DiffusionConfig:
    timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    schedule: str = "linear"


@dataclass
class TrainConfig:
    epochs: int = 100
    learning_rate: float = 2e-4
    gradient_clip: float = 1.0
    save_every: int = 10
    sample_every: int = 5
    n_samples: int = 64
    checkpoint_dir: str = "./results/checkpoints"
    results_dir: str = "./results"
    profile_steps: int = 0  # 0 = disabled, N = profile first N steps


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    device: str = "cuda"
    seed: int = 42

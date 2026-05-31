import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from config.config import DataConfig


def get_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),  # map [0,1] to [-1,1]
    ])


def get_dataloader(cfg: DataConfig, pin_memory: bool | None = None) -> DataLoader:
    """Build an MNIST DataLoader.

    pin_memory controls whether host tensors are page-locked before transfer
    to the GPU. When None, uses cfg.pin_memory.
    """
    use_pin = cfg.pin_memory if pin_memory is None else pin_memory

    dataset = datasets.MNIST(
        root=cfg.data_dir,
        train=True,
        download=True,
        transform=get_transform(cfg.image_size),
    )

    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=use_pin,
        drop_last=True,
        persistent_workers=cfg.num_workers > 0,
    )

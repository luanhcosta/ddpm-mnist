import csv
import json
import time
from pathlib import Path


class MetricsLogger:
    """Lightweight CSV logger for training metrics."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fields: list[str] | None = None
        self._file = None
        self._writer = None

    def log(self, **kwargs) -> None:
        if self._fields is None:
            self._fields = list(kwargs.keys())
            self._file = open(self.path, "w", newline="")
            self._writer = csv.DictWriter(self._file, fieldnames=self._fields)
            self._writer.writeheader()
        self._writer.writerow(kwargs)
        self._file.flush()

    def close(self) -> None:
        if self._file:
            self._file.close()


class GPUMemoryTracker:
    """Tracks peak GPU memory usage across training."""

    def __init__(self, device: str):
        self.device = device

    def snapshot(self) -> dict:
        import torch
        return {
            "allocated_mb": round(torch.cuda.memory_allocated(self.device) / 1e6, 1),
            "reserved_mb": round(torch.cuda.memory_reserved(self.device) / 1e6, 1),
            "peak_allocated_mb": round(torch.cuda.max_memory_allocated(self.device) / 1e6, 1),
        }

    def reset_peak(self) -> None:
        import torch
        torch.cuda.reset_peak_memory_stats(self.device)

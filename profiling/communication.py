"""CPU-GPU communication benchmarks.

This module measures the concrete cost of every point where the CPU and GPU
exchange data or control during DDPM training and inference.

Benchmarks:
    1. H2D Transfer Bandwidth   — pageable vs pinned memory, multiple tensor sizes
    2. DataLoader Overhead       — pin_memory=True vs False during training batches
    3. Noise Schedule Transfer   — cost of moving T precomputed tensors to GPU
    4. Sampling Loop Sync        — 1000-step DDPM with forced syncs vs pure-GPU vs DDIM-50
    5. Training Step Profile     — torch.profiler chrome trace of one training step
"""

import json
import time
from pathlib import Path

import torch
import torch.nn as nn

from config.config import Config
from data.dataset import get_dataloader
from models.diffusion import DiffusionProcess, make_beta_schedule
from models.unet import UNet
from sampling.sampler import ddpm_sample_naive_sync, ddpm_sample_no_sync, ddim_sample


def _sync_time(fn) -> float:
    """Run fn(), return wall-clock seconds with proper GPU synchronization."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return time.perf_counter() - t0


# ── 1. H2D Transfer Bandwidth ────────────────────────────────────────────────

def benchmark_h2d_transfer(
    device: torch.device,
    sizes_mb: list[float] | None = None,
    n_runs: int = 30,
) -> dict:
    """Measure host-to-device bandwidth for pageable vs pinned memory.

    Key insight: pinned (page-locked) memory bypasses the OS page table and
    allows the DMA engine to transfer directly, avoiding an intermediate
    CPU-side copy. The speedup is most visible for large tensors.
    """
    if sizes_mb is None:
        sizes_mb = [1, 10, 50, 100, 500]

    results = []
    for mb in sizes_mb:
        n_elements = int(mb * 1024 * 1024 / 4)  # float32 = 4 bytes

        # Pageable memory (standard malloc)
        pageable = torch.randn(n_elements)
        t_pageable = []
        for _ in range(n_runs):
            t_pageable.append(_sync_time(lambda: pageable.to(device, non_blocking=False)))
        bw_pageable = mb / (sum(t_pageable) / n_runs) / 1024  # GB/s

        # Pinned memory (cudaMallocHost)
        pinned = torch.randn(n_elements).pin_memory()
        t_pinned = []
        gpu_buf = torch.empty(n_elements, device=device)
        for _ in range(n_runs):
            t_pinned.append(_sync_time(lambda: gpu_buf.copy_(pinned, non_blocking=False)))
        bw_pinned = mb / (sum(t_pinned) / n_runs) / 1024  # GB/s

        results.append({
            "size_mb": mb,
            "pageable_ms": round(sum(t_pageable) / n_runs * 1000, 2),
            "pinned_ms": round(sum(t_pinned) / n_runs * 1000, 2),
            "pageable_gbps": round(bw_pageable, 2),
            "pinned_gbps": round(bw_pinned, 2),
            "speedup": round(bw_pinned / bw_pageable, 2),
        })

    return {"h2d_transfer": results}


# ── 2. DataLoader pin_memory ──────────────────────────────────────────────────

def benchmark_dataloader(
    cfg: Config,
    device: torch.device,
    n_batches: int = 50,
) -> dict:
    """Compare batch loading time with and without pin_memory.

    pin_memory=True pre-allocates host tensors in page-locked memory so the
    DataLoader's background workers can start the DMA transfer immediately
    when the main thread calls .to(device), without waiting for the OS to
    pin the pages first.
    """
    rows = []
    for pin in (False, True):
        loader = get_dataloader(cfg.data, pin_memory=pin)
        it = iter(loader)

        # Warm up (first batch may trigger MNIST download + caching)
        batch, _ = next(it)
        batch.to(device)

        times = []
        for _ in range(n_batches):
            try:
                batch, _ = next(it)
            except StopIteration:
                it = iter(loader)
                batch, _ = next(it)
            t = _sync_time(lambda: batch.to(device, non_blocking=pin))
            times.append(t)

        avg_ms = sum(times) / len(times) * 1000
        rows.append({
            "pin_memory": pin,
            "avg_ms_per_batch": round(avg_ms, 3),
            "batches_per_second": round(1000 / avg_ms, 1),
        })

    speedup = rows[0]["avg_ms_per_batch"] / rows[1]["avg_ms_per_batch"]
    return {
        "dataloader": rows,
        "pin_memory_speedup": round(speedup, 2),
    }


# ── 3. Noise Schedule Transfer ────────────────────────────────────────────────

def benchmark_noise_schedule(
    cfg: Config,
    device: torch.device,
    n_runs: int = 50,
) -> dict:
    """Measure the cost of precomputing the noise schedule on CPU vs GPU.

    In DiffusionProcess.__init__, we compute all T schedule tensors on the CPU
    (using torch.linspace, torch.cumprod, etc.) and then .to(device) them.
    The alternative is computing them directly on the GPU.

    This benchmark separates computation time from transfer time.
    """
    T = cfg.diffusion.timesteps

    # Strategy A: compute on CPU, transfer to GPU (what we do in DiffusionProcess)
    def strategy_cpu_then_transfer():
        betas = make_beta_schedule(T, cfg.diffusion.beta_start, cfg.diffusion.beta_end, cfg.diffusion.schedule)
        alphas = 1.0 - betas
        abar = torch.cumprod(alphas, 0)
        # Transfer 6 tensors of size T
        for t in [betas, alphas, abar, abar.sqrt(), (1 - abar).sqrt(), (1.0 / alphas).sqrt()]:
            t.to(device)

    # Strategy B: compute directly on GPU
    def strategy_gpu_direct():
        betas = make_beta_schedule(T, cfg.diffusion.beta_start, cfg.diffusion.beta_end, cfg.diffusion.schedule).to(device)
        alphas = 1.0 - betas
        abar = torch.cumprod(alphas, 0)
        _ = abar.sqrt()
        _ = (1 - abar).sqrt()
        _ = (1.0 / alphas).sqrt()

    # Warmup
    strategy_cpu_then_transfer()
    strategy_gpu_direct()

    t_cpu_transfer = [_sync_time(strategy_cpu_then_transfer) for _ in range(n_runs)]
    t_gpu_direct = [_sync_time(strategy_gpu_direct) for _ in range(n_runs)]

    avg_cpu = sum(t_cpu_transfer) / n_runs * 1000
    avg_gpu = sum(t_gpu_direct) / n_runs * 1000

    return {
        "noise_schedule": {
            "T": T,
            "cpu_compute_then_transfer_ms": round(avg_cpu, 3),
            "gpu_direct_ms": round(avg_gpu, 3),
            "note": "cpu+transfer is slower due to H2D overhead on small tensors",
        }
    }


# ── 4. Sampling Loop Sync Cost ────────────────────────────────────────────────

def benchmark_sampling_loop(
    model: UNet,
    diffusion: DiffusionProcess,
    device: torch.device,
    n_samples: int = 4,
    n_runs: int = 3,
) -> dict:
    """Compare three sampling strategies for total wall-clock time.

    naive_sync   : DDPM-1000 with .item() at every step → 1000 CPU-GPU syncs
    pure_gpu     : DDPM-1000 with one sync at the end
    ddim_50      : DDIM with 50 steps, deterministic, one sync

    The sync overhead in the naive case adds up: each .item() blocks the CPU
    until that step's kernels complete, preventing the GPU from building up a
    queue of work. The pure_gpu version lets the GPU run ahead continuously.
    """
    rows = []

    for _ in range(n_runs):
        _, t_naive = ddpm_sample_naive_sync(model, diffusion, n_samples, device)
        _, t_pure = ddpm_sample_no_sync(model, diffusion, n_samples, device)
        _, t_ddim = ddim_sample(model, diffusion, n_samples, device, n_steps=50)

        rows.append({
            "ddpm_naive_sync_s": round(t_naive, 3),
            "ddpm_pure_gpu_s": round(t_pure, 3),
            "ddim_50_s": round(t_ddim, 3),
        })

    avg = lambda key: round(sum(r[key] for r in rows) / n_runs, 3)
    return {
        "sampling_loop": {
            "n_samples": n_samples,
            "n_runs": n_runs,
            "ddpm_1000_naive_sync_s": avg("ddpm_naive_sync_s"),
            "ddpm_1000_pure_gpu_s": avg("ddpm_pure_gpu_s"),
            "ddim_50_s": avg("ddim_50_s"),
            "sync_overhead_factor": round(avg("ddpm_naive_sync_s") / avg("ddpm_pure_gpu_s"), 2),
            "ddim_speedup_vs_ddpm": round(avg("ddpm_pure_gpu_s") / avg("ddim_50_s"), 1),
        }
    }


# ── 5. Training Step Profile ──────────────────────────────────────────────────

def profile_training_step(
    model: UNet,
    diffusion: DiffusionProcess,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    output_dir: Path,
    n_steps: int = 5,
) -> None:
    """Record a torch.profiler chrome trace for N training steps.

    The trace file can be opened in Chrome at chrome://tracing or in
    TensorBoard's profiler tab. It shows:
      - CPU-side Python dispatch time per kernel launch
      - CUDA kernel execution time on GPU
      - cudaMemcpyAsync for each batch (H2D transfer)
      - Synchronization points

    This makes the CPU-GPU communication pattern *visible* rather than
    just a number.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    model.train()

    data_iter = iter(dataloader)

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=n_steps),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(output_dir / "profiler_trace")),
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        for step in range(n_steps + 2):  # +2 for wait+warmup
            try:
                x, _ = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                x, _ = next(data_iter)

            x = x.to(device, non_blocking=True)
            t = torch.randint(0, diffusion.T, (x.size(0),), device=device)

            optimizer.zero_grad()
            loss = diffusion.loss(model, x, t)
            loss.backward()
            optimizer.step()

            prof.step()

    # Also export as standalone chrome trace JSON
    prof.export_chrome_trace(str(output_dir / "chrome_trace.json"))
    print(f"  Chrome trace saved to {output_dir}/chrome_trace.json")
    print(f"  TensorBoard trace saved to {output_dir}/profiler_trace/")

    # Print top-20 ops by CUDA time
    print("\n  Top CUDA kernels (by self CUDA time):")
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20))


# ── 6. Training Step Timing Breakdown ────────────────────────────────────────

def benchmark_training_step(
    model: UNet,
    diffusion: DiffusionProcess,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    n_steps: int = 100,
) -> dict:
    """Break down each phase of a training step.

    Phases timed separately:
      - data_load_h2d: time to load batch from DataLoader + copy to GPU
      - noise_sample:  torch.randn_like (GPU-only, no H2D)
      - forward:       model(xt, t)
      - loss:          F.mse_loss
      - backward:      loss.backward()
      - optimizer:     optimizer.step()

    torch.cuda.synchronize() is called before each timer to ensure we measure
    actual GPU execution time, not just kernel dispatch latency.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    model.train()

    phase_times = {k: [] for k in ["data_h2d", "noise_t_sample", "forward", "loss", "backward", "optimizer"]}

    data_iter = iter(dataloader)

    # Warmup
    for _ in range(5):
        try:
            x, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            x, _ = next(data_iter)
        x = x.to(device)
        t = torch.randint(0, diffusion.T, (x.size(0),), device=device)
        loss = diffusion.loss(model, x, t)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    for _ in range(n_steps):
        # Phase 1: data loading + H2D transfer
        try:
            cpu_batch, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            cpu_batch, _ = next(data_iter)
        t_h2d = _sync_time(lambda: cpu_batch.to(device, non_blocking=False))
        x = cpu_batch.to(device)

        # Phase 2: sample noise + timestep (pure GPU ops, no H2D)
        def sample_noise_t():
            nonlocal noise, ts
            noise = torch.randn_like(x)
            ts = torch.randint(0, diffusion.T, (x.size(0),), device=device)
        noise, ts = None, None
        t_noise = _sync_time(sample_noise_t)

        # Phase 3: forward pass
        xt = diffusion.q_sample(x, ts, noise)
        forward_out = [None]
        t_fwd = _sync_time(lambda: forward_out.__setitem__(0, model(xt, ts)))
        pred = forward_out[0]

        # Phase 4: loss computation
        import torch.nn.functional as F
        loss_val = [None]
        t_loss = _sync_time(lambda: loss_val.__setitem__(0, F.mse_loss(pred, noise)))

        # Phase 5: backward
        optimizer.zero_grad()
        t_bwd = _sync_time(lambda: loss_val[0].backward())

        # Phase 6: optimizer step
        t_opt = _sync_time(optimizer.step)

        phase_times["data_h2d"].append(t_h2d * 1000)
        phase_times["noise_t_sample"].append(t_noise * 1000)
        phase_times["forward"].append(t_fwd * 1000)
        phase_times["loss"].append(t_loss * 1000)
        phase_times["backward"].append(t_bwd * 1000)
        phase_times["optimizer"].append(t_opt * 1000)

    averages = {k: round(sum(v) / len(v), 3) for k, v in phase_times.items()}
    total = sum(averages.values())

    return {
        "training_step_breakdown": {
            "n_steps": n_steps,
            "phases_avg_ms": averages,
            "total_ms": round(total, 3),
            "data_h2d_pct": round(averages["data_h2d"] / total * 100, 1),
        }
    }


# ── Entry point ───────────────────────────────────────────────────────────────

def run_all_benchmarks(
    cfg: Config,
    model: UNet,
    diffusion: DiffusionProcess,
    device: torch.device,
    output_dir: Path,
) -> dict:
    """Run all benchmarks, print a summary, and save results to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    print("\n[1/6] H2D Transfer Bandwidth...")
    results.update(benchmark_h2d_transfer(device))

    print("[2/6] DataLoader pin_memory overhead...")
    dataloader = get_dataloader(cfg.data)
    results.update(benchmark_dataloader(cfg, device))

    print("[3/6] Noise schedule precomputation...")
    results.update(benchmark_noise_schedule(cfg, device))

    print("[4/6] Sampling loop sync overhead...")
    results.update(benchmark_sampling_loop(model, diffusion, device))

    print("[5/6] Training step breakdown...")
    results.update(benchmark_training_step(model, diffusion, dataloader, device))

    print("[6/6] torch.profiler chrome trace...")
    profile_training_step(model, diffusion, dataloader, device, output_dir)

    out_path = output_dir / "benchmarks.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll results saved to {out_path}")

    _print_summary(results)
    return results


def _print_summary(results: dict) -> None:
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)

    if "h2d_transfer" in results:
        print("\n── H2D Transfer Bandwidth ──────────────────────────────────")
        print(f"{'Size':>8}  {'Pageable':>12}  {'Pinned':>12}  {'Speedup':>8}")
        for r in results["h2d_transfer"]:
            print(f"{r['size_mb']:>7}MB  {r['pageable_gbps']:>10.2f}GB/s  {r['pinned_gbps']:>10.2f}GB/s  {r['speedup']:>7.2f}x")

    if "dataloader" in results:
        print("\n── DataLoader Overhead ──────────────────────────────────────")
        for r in results["dataloader"]:
            label = "pin=True " if r["pin_memory"] else "pin=False"
            print(f"  {label}: {r['avg_ms_per_batch']:.3f} ms/batch  ({r['batches_per_second']:.1f} batches/s)")
        print(f"  Speedup: {results['pin_memory_speedup']:.2f}x")

    if "sampling_loop" in results:
        s = results["sampling_loop"]
        print("\n── Sampling Loop Sync Overhead ──────────────────────────────")
        print(f"  DDPM-1000 (naive sync):  {s['ddpm_1000_naive_sync_s']:.3f}s")
        print(f"  DDPM-1000 (pure GPU):    {s['ddpm_1000_pure_gpu_s']:.3f}s")
        print(f"  DDIM-50   (pure GPU):    {s['ddim_50_s']:.3f}s")
        print(f"  Sync overhead:   {s['sync_overhead_factor']:.2f}x slower")
        print(f"  DDIM speedup:    {s['ddim_speedup_vs_ddpm']:.1f}x vs DDPM")

    if "training_step_breakdown" in results:
        b = results["training_step_breakdown"]
        print("\n── Training Step Breakdown ──────────────────────────────────")
        for phase, ms in b["phases_avg_ms"].items():
            pct = ms / b["total_ms"] * 100
            bar = "█" * int(pct / 2)
            print(f"  {phase:20s} {ms:7.3f} ms  {pct:5.1f}%  {bar}")
        print(f"  {'TOTAL':20s} {b['total_ms']:7.3f} ms")
        print(f"  H2D is {b['data_h2d_pct']:.1f}% of total step time")

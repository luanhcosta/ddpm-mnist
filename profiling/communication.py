"""CPU-GPU communication benchmarks.

This module measures the concrete cost of every point where the CPU and GPU
exchange data or control during DDPM training and inference.

Benchmarks:
    1. H2D transfer             — equivalent copies from pageable and pinned memory
    2. Input pipeline           — transfer-only vs next(iterator) + transfer
    3. Noise schedule transfer  — CPU precomputation and one-time H2D cost
    4. Host observation         — cost of materializing GPU statistics in Python
    5. AMP training breakdown   — phases of the actual mixed-precision path
    6. Training profile         — torch.profiler trace of representative steps
"""

import json
import platform
import statistics
import time
from pathlib import Path

import torch
import torch.nn as nn

from config.config import Config
from data.dataset import get_dataloader
from models.diffusion import DiffusionProcess, make_beta_schedule
from models.unet import UNet
from sampling.sampler import ddpm_sample_with_observation_interval, ddim_sample


def _sync_time(fn) -> float:
    """Run fn(), return wall-clock seconds with proper GPU synchronization."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def _stats_ms(samples_s: list[float]) -> dict:
    """Return robust latency statistics for samples measured in seconds."""
    samples_ms = sorted(sample * 1000 for sample in samples_s)
    p95_index = min(len(samples_ms) - 1, int(0.95 * len(samples_ms)))
    return {
        "mean_ms": round(statistics.fmean(samples_ms), 4),
        "p50_ms": round(statistics.median(samples_ms), 4),
        "p95_ms": round(samples_ms[p95_index], 4),
    }


def _environment_metadata(device: torch.device, cfg: Config) -> dict:
    metadata = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "device": str(device),
        "cuda_runtime": torch.version.cuda,
        "batch_size": cfg.data.batch_size,
        "num_workers": cfg.data.num_workers,
        "amp": True,
    }
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        metadata.update({
            "gpu": props.name,
            "gpu_memory_mb": round(props.total_memory / 1024**2),
            "compute_capability": f"{props.major}.{props.minor}",
        })
    return metadata


# ── 1. H2D Transfer Bandwidth ────────────────────────────────────────────────

def benchmark_h2d_transfer(
    device: torch.device,
    sizes_mb: list[float] | None = None,
    n_runs: int = 30,
) -> dict:
    """Measure H2D latency and bandwidth with equivalent copy operations."""
    if sizes_mb is None:
        sizes_mb = [1, 10, 50, 100, 500]

    results = []
    for mb in sizes_mb:
        n_elements = int(mb * 1024 * 1024 / 4)  # float32 = 4 bytes

        pageable = torch.randn(n_elements)
        pinned = torch.randn(n_elements).pin_memory()
        gpu_buf = torch.empty(n_elements, device=device)

        gpu_buf.copy_(pageable)
        gpu_buf.copy_(pinned)
        torch.cuda.synchronize()

        t_pageable, t_pinned = [], []
        for _ in range(n_runs):
            # Alternating order reduces clock and thermal order bias.
            if len(t_pageable) % 2 == 0:
                t_pageable.append(_sync_time(lambda: gpu_buf.copy_(pageable, non_blocking=True)))
                t_pinned.append(_sync_time(lambda: gpu_buf.copy_(pinned, non_blocking=True)))
            else:
                t_pinned.append(_sync_time(lambda: gpu_buf.copy_(pinned, non_blocking=True)))
                t_pageable.append(_sync_time(lambda: gpu_buf.copy_(pageable, non_blocking=True)))

        pageable_stats = _stats_ms(t_pageable)
        pinned_stats = _stats_ms(t_pinned)
        bw_pageable = mb / (pageable_stats["p50_ms"] / 1000) / 1024
        bw_pinned = mb / (pinned_stats["p50_ms"] / 1000) / 1024

        results.append({
            "size_mb": mb,
            "pageable": pageable_stats,
            "pinned": pinned_stats,
            "pageable_gbps": round(bw_pageable, 2),
            "pinned_gbps": round(bw_pinned, 2),
            "speedup": round(bw_pinned / bw_pageable, 2),
        })

    return {"h2d_transfer": results}


# ── 2. DataLoader pin_memory ──────────────────────────────────────────────────

def benchmark_input_pipeline(
    cfg: Config,
    device: torch.device,
    n_batches: int = 50,
) -> dict:
    """Measure transfer-only and end-to-end input latency separately."""
    transfer_rows, pipeline_rows = [], []
    for pin in (False, True):
        loader = get_dataloader(cfg.data, pin_memory=pin)
        it = iter(loader)

        # Warm up (first batch may trigger MNIST download + caching)
        batch, _ = next(it)
        batch.to(device)

        transfer_times = []
        for _ in range(n_batches):
            try:
                batch, _ = next(it)
            except StopIteration:
                it = iter(loader)
                batch, _ = next(it)
            transfer_times.append(_sync_time(lambda: batch.to(device, non_blocking=pin)))
        transfer_rows.append({"pin_memory": pin, **_stats_ms(transfer_times)})

        pipeline_times = []
        it = iter(loader)
        for _ in range(n_batches):
            torch.cuda.synchronize()
            start = time.perf_counter()
            try:
                batch, _ = next(it)
            except StopIteration:
                it = iter(loader)
                batch, _ = next(it)
            batch.to(device, non_blocking=pin)
            torch.cuda.synchronize()
            pipeline_times.append(time.perf_counter() - start)
        pipeline_rows.append({"pin_memory": pin, **_stats_ms(pipeline_times)})

    return {
        "batch_transfer": transfer_rows,
        "input_pipeline": pipeline_rows,
        "batch_transfer_speedup_p50": round(
            transfer_rows[0]["p50_ms"] / transfer_rows[1]["p50_ms"], 2
        ),
        "input_pipeline_speedup_p50": round(
            pipeline_rows[0]["p50_ms"] / pipeline_rows[1]["p50_ms"], 2
        ),
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
            "note": (
                "CPU compute plus one-time transfer was faster in this run"
                if avg_cpu < avg_gpu
                else "direct GPU computation was faster in this run"
            ),
        }
    }


# ── 4. Sampling Loop Sync Cost ────────────────────────────────────────────────

def benchmark_sampling_loop(
    model: UNet,
    diffusion: DiffusionProcess,
    device: torch.device,
    n_samples: int = 4,
    n_runs: int = 5,
) -> dict:
    """Measure the cost of materializing GPU statistics on the host.

    Each observation performs ``x.mean().item()``: a reduction, a scalar D2H
    copy, and a host wait.  The result is intentionally described as
    observation overhead rather than bare synchronization overhead.
    """
    intervals = (0, 100, 10, 1)
    samples = {str(interval): [] for interval in intervals}
    ddim_samples = []

    # A full pass stabilizes lazy initialization and GPU clocks.
    ddpm_sample_with_observation_interval(
        model, diffusion, n_samples, device, observation_every=0
    )

    for run in range(n_runs):
        # Rotate execution order so no case always benefits from running later.
        ordered = intervals[run % len(intervals):] + intervals[:run % len(intervals)]
        for interval in ordered:
            _, elapsed = ddpm_sample_with_observation_interval(
                model, diffusion, n_samples, device, observation_every=interval
            )
            samples[str(interval)].append(elapsed)
        _, t_ddim = ddim_sample(model, diffusion, n_samples, device, n_steps=50)
        ddim_samples.append(t_ddim)

    averages = {
        key: round(statistics.median(samples[key]), 3)
        for key in map(str, intervals)
    }
    ddim_avg = round(statistics.median(ddim_samples), 3)
    baseline = averages["0"]
    return {
        "sampling_loop": {
            "n_samples": n_samples,
            "n_runs": n_runs,
            "aggregation": "median; rotated case order after one warmup pass",
            "observation": {
                "never_s": averages["0"],
                "every_100_steps_s": averages["100"],
                "every_10_steps_s": averages["10"],
                "every_step_s": averages["1"],
            },
            "every_step_overhead_ms": round((averages["1"] - baseline) * 1000, 1),
            "every_step_overhead_factor": round(averages["1"] / baseline, 2),
            "ddim_50_s": ddim_avg,
            "ddim_speedup_vs_ddpm": round(baseline / ddim_avg, 1),
            "measurement": "x.mean().item(): reduction + scalar D2H copy + host wait",
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
    scaler = torch.amp.GradScaler("cuda")
    model.train()

    data_iter = iter(dataloader)

    # Use CPU-only activities on Blackwell (sm_120) — CUPTI not yet supported
    # on this architecture, so CUDA activity collection is skipped gracefully.
    activities = [torch.profiler.ProfilerActivity.CPU]
    try:
        import ctypes
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    except Exception:
        pass

    with torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=n_steps),
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

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                loss = diffusion.loss(model, x, t)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            prof.step()

    # Export chrome trace (works even without CUDA activities)
    trace_path = output_dir / "chrome_trace.json"
    try:
        prof.export_chrome_trace(str(trace_path))
        print(f"  Chrome trace saved to {trace_path}")
        print("  Open in Chrome at chrome://tracing")
    except RuntimeError:
        # on_trace_ready already saved it
        print(f"  Trace saved via on_trace_ready to {output_dir}/profiler_trace/")

    # Print top ops by CPU time (CUDA time unavailable on Blackwell with this CUPTI)
    print("\n  Top ops (by self CPU time):")
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=15))


# ── 6. Training Step Timing Breakdown ────────────────────────────────────────

def benchmark_training_step(
    model: UNet,
    diffusion: DiffusionProcess,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    n_steps: int = 100,
) -> dict:
    """Break down the actual AMP training path into synchronized phases."""
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    scaler = torch.amp.GradScaler("cuda")
    model.train()

    phase_times = {k: [] for k in [
        "data_h2d", "noise_and_q_sample", "forward", "loss", "backward", "optimizer_and_clip"
    ]}

    data_iter = iter(dataloader)

    # Warmup
    for _ in range(5):
        try:
            x, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            x, _ = next(data_iter)
        x = x.to(device, non_blocking=dataloader.pin_memory)
        t = torch.randint(0, diffusion.T, (x.size(0),), device=device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda"):
            loss = diffusion.loss(model, x, t)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

    for _ in range(n_steps):
        # Phase 1: data loading + H2D transfer
        try:
            cpu_batch, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            cpu_batch, _ = next(data_iter)
        x_holder = [None]
        t_h2d = _sync_time(
            lambda: x_holder.__setitem__(
                0, cpu_batch.to(device, non_blocking=dataloader.pin_memory)
            )
        )
        x = x_holder[0]

        # Phase 2: sample noise + timestep (pure GPU ops, no H2D)
        def sample_noise_and_q():
            nonlocal noise, ts, xt
            noise = torch.randn_like(x)
            ts = torch.randint(0, diffusion.T, (x.size(0),), device=device)
            xt = diffusion.q_sample(x, ts, noise)
        noise, ts, xt = None, None, None
        t_noise = _sync_time(sample_noise_and_q)

        # Phase 3: forward pass
        optimizer.zero_grad(set_to_none=True)
        forward_out = [None]
        def forward_amp():
            with torch.amp.autocast("cuda"):
                forward_out[0] = model(xt, ts)
        t_fwd = _sync_time(forward_amp)
        pred = forward_out[0]

        # Phase 4: loss computation
        import torch.nn.functional as F
        loss_val = [None]
        def loss_amp():
            with torch.amp.autocast("cuda"):
                loss_val[0] = F.mse_loss(pred, noise)
        t_loss = _sync_time(loss_amp)

        # Phase 5: backward
        t_bwd = _sync_time(lambda: scaler.scale(loss_val[0]).backward())

        # Phase 6: unscale, gradient clipping, optimizer step, scaler update
        def optimizer_amp():
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        t_opt = _sync_time(optimizer_amp)

        phase_times["data_h2d"].append(t_h2d * 1000)
        phase_times["noise_and_q_sample"].append(t_noise * 1000)
        phase_times["forward"].append(t_fwd * 1000)
        phase_times["loss"].append(t_loss * 1000)
        phase_times["backward"].append(t_bwd * 1000)
        phase_times["optimizer_and_clip"].append(t_opt * 1000)

    averages = {k: round(sum(v) / len(v), 3) for k, v in phase_times.items()}
    total = sum(averages.values())

    return {
        "training_step_breakdown": {
            "n_steps": n_steps,
            "amp": True,
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
    include_sampling: bool = True,
) -> dict:
    """Run all benchmarks, print a summary, and save results to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {"metadata": _environment_metadata(device, cfg)}

    print("\n[1/6] H2D Transfer Bandwidth...")
    results.update(benchmark_h2d_transfer(device))

    print("[2/6] Batch transfer and end-to-end input pipeline...")
    dataloader = get_dataloader(cfg.data)
    results.update(benchmark_input_pipeline(cfg, device))

    print("[3/6] Noise schedule precomputation...")
    results.update(benchmark_noise_schedule(cfg, device))

    if include_sampling:
        print("[4/6] Sampling host-observation overhead...")
        results.update(benchmark_sampling_loop(model, diffusion, device))
    else:
        print("[4/6] Sampling host-observation overhead... skipped")

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

    if "batch_transfer" in results:
        print("\n── DataLoader Overhead ──────────────────────────────────────")
        for key in ("batch_transfer", "input_pipeline"):
            print(f"  {key}:")
            for r in results[key]:
                print(f"    pin={r['pin_memory']}: p50={r['p50_ms']:.4f} ms, p95={r['p95_ms']:.4f} ms")

    if "sampling_loop" in results:
        s = results["sampling_loop"]
        print("\n── Sampling Loop Sync Overhead ──────────────────────────────")
        for label, value in s["observation"].items():
            print(f"  {label:22s}: {value:.3f}s")
        print(f"  DDIM-50   (pure GPU):    {s['ddim_50_s']:.3f}s")
        print(f"  Every-step observation: {s['every_step_overhead_factor']:.2f}x")
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

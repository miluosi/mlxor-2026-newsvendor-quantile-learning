"""Benchmark full-graph and screen-and-replay RSETO-IPA on one GPU."""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from model.rseto_ipa_spline import (
    RSETOIPASplineNewsvendor,
    screen_selected_base_noise,
)


def resolve_device(requested):
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def time_steps(step, repeats, warmup, device):
    for _ in range(warmup):
        step()
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            step()
        end.record()
        synchronize(device)
        elapsed = start.elapsed_time(end) / (1000.0 * repeats)
        peak_memory = torch.cuda.max_memory_allocated(device)
    else:
        start = time.perf_counter()
        for _ in range(repeats):
            step()
        synchronize(device)
        elapsed = (time.perf_counter() - start) / repeats
        peak_memory = None
    return elapsed, peak_memory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--dim", type=int, default=24)
    parser.add_argument("--replications", type=int, default=16)
    parser.add_argument("--m", type=int, default=512)
    parser.add_argument("--max-simulation-values", type=int, default=1048576)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--profile-trace", type=Path)
    args = parser.parse_args()
    if min(
        args.batch_size,
        args.dim,
        args.replications,
        args.m,
        args.max_simulation_values,
        args.repeats,
    ) < 1:
        parser.error("Batch, dimensions, simulation sizes, and repeats must be positive.")

    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    model_kwargs = {
        "targetdim": 1,
        "labeldim": args.dim,
        "latent": 1,
        "data_len": args.batch_size,
        "epoch": 1,
        "target_quantile": 0.7,
        "cost_under": 7.0,
        "cost_over": 3.0,
        "num_transforms": 4,
        "num_bins": 16,
        "hidden_dim": 64,
        "hidden_layers": 2,
    }
    full_graph = RSETOIPASplineNewsvendor(**model_kwargs).to(device).eval()
    screen_replay = RSETOIPASplineNewsvendor(**model_kwargs).to(device).eval()
    screen_replay.load_state_dict(copy.deepcopy(full_graph.state_dict()), strict=True)
    condition = torch.randn(args.batch_size, args.dim, device=device)
    demand = torch.randn(args.batch_size, 1, device=device)
    base_noise = torch.randn(
        args.batch_size,
        args.replications,
        args.m,
        1,
        device=device,
    )

    def full_graph_step():
        full_graph.zero_grad(set_to_none=True)
        loss, _ = full_graph.rseto_ipa_objective(
            condition,
            demand,
            replications=args.replications,
            samples_per_replication=args.m,
            smoothing_mu=0.05,
            fidelity_weight=0.5,
            base_noise=base_noise,
        )
        loss.backward()

    def screen_replay_step():
        screen_replay.zero_grad(set_to_none=True)
        selected_noise, _, _ = screen_selected_base_noise(
            screen_replay.backbone,
            condition,
            replications=args.replications,
            samples_per_replication=args.m,
            target_quantile=screen_replay.target_quantile,
            max_simulation_values=args.max_simulation_values,
            base_noise=base_noise,
        )
        loss, _ = screen_replay.rseto_ipa_replay_objective(
            condition,
            demand,
            selected_noise=selected_noise,
            smoothing_mu=0.05,
            fidelity_weight=0.5,
        )
        loss.backward()

    full_graph_seconds, full_graph_memory = time_steps(
        full_graph_step,
        args.repeats,
        args.warmup,
        device,
    )
    screen_replay_seconds, screen_replay_memory = time_steps(
        screen_replay_step,
        args.repeats,
        args.warmup,
        device,
    )
    result = {
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else str(device)
        ),
        "batch_size": args.batch_size,
        "replications": args.replications,
        "m": args.m,
        "simulated_values_per_step": args.batch_size * args.replications * args.m,
        "full_graph_seconds_per_step": full_graph_seconds,
        "screen_replay_seconds_per_step": screen_replay_seconds,
        "speedup": full_graph_seconds / screen_replay_seconds,
        "screening_values_per_second": (
            args.batch_size
            * args.replications
            * args.m
            / screen_replay_seconds
        ),
        "full_graph_peak_memory_bytes": full_graph_memory,
        "screen_replay_peak_memory_bytes": screen_replay_memory,
    }
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if args.profile_trace is not None:
        activities = [ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(ProfilerActivity.CUDA)
        with profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
        ) as profiler:
            with record_function("rseto_screen_replay_step"):
                screen_replay_step()
        synchronize(device)
        args.profile_trace.parent.mkdir(parents=True, exist_ok=True)
        profiler.export_chrome_trace(str(args.profile_trace))


if __name__ == "__main__":
    main()

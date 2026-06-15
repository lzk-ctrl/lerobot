#!/usr/bin/env python

from __future__ import annotations

import argparse
import copy
import statistics
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.profile_modeling_pi05 import PI05Policy
from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors
from lerobot.policies.utils import prepare_observation_for_inference

from pi05_runtime import build_pi05_observation_from_robot_observer_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile the PI0.5 inference path")
    parser.add_argument("--model-id", default="lerobot/pi05_base", help="HF model id or local path")
    parser.add_argument(
        "--device",
        default="auto",
        help="Target inference device, e.g. auto, cpu, mps, cuda, cuda:1",
    )
    parser.add_argument("--task", default="walk forward while avoiding all obstacles")
    parser.add_argument("--state-dim", type=int, default=22)
    parser.add_argument("--action-dim", type=int, default=3)
    parser.add_argument("--image-height", type=int, default=480)
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument(
        "--observation-json",
        default=None,
        help="Optional robot_observer JSON snapshot. If omitted, a dummy zero observation is used.",
    )
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument(
        "--num-action-samples",
        type=int,
        default=1,
        help="Number of noise samples/action chunks to infer per observation.",
    )
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument(
        "--tokenizer-path",
        default=None,
        help="Optional local tokenizer directory. Defaults to <model-id>/tokenizer when present.",
    )
    parser.add_argument(
        "--actions-per-chunk",
        type=int,
        default=1,
        help="Requested actions per chunk. The model still runs full chunk inference before slicing.",
    )
    parser.add_argument("--warmup-runs", type=int, default=3)
    parser.add_argument("--profile-runs", type=int, default=5)
    parser.add_argument(
        "--benchmark-steps",
        default=None,
        help="Comma-separated num_inference_steps sweep, e.g. 4,6,8,10. Runs latency benchmark mode.",
    )
    parser.add_argument(
        "--torch-profiler",
        action="store_true",
        help="Enable torch.profiler and export a trace directory for TensorBoard/Chrome tracing.",
    )
    parser.add_argument(
        "--trace-dir",
        default="/tmp/pi05_torch_profile",
        help="Directory used by torch.profiler trace handler.",
    )
    parser.add_argument("--record-shapes", action="store_true")
    parser.add_argument("--profile-memory", action="store_true")
    parser.add_argument("--row-limit", type=int, default=30)
    parser.add_argument(
        "--sort-by",
        default=None,
        help="torch.profiler table sort key. Defaults to self_cuda_time_total on CUDA, else self_cpu_time_total.",
    )
    return parser.parse_args()


def select_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        device = torch.device(device_arg)
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but is not available in this environment")
            if device.index is not None:
                torch.cuda.set_device(device.index)
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synchronize_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def resolve_tokenizer_path(model_id: str, tokenizer_path: str | None) -> str:
    if tokenizer_path:
        return tokenizer_path

    model_path = Path(model_id)
    local_tokenizer_dir = model_path / "tokenizer"
    if model_path.exists() and local_tokenizer_dir.is_dir():
        return str(local_tokenizer_dir)

    return "google/paligemma-3b-pt-224"


def build_config(args: argparse.Namespace, device: torch.device) -> PI05Config:
    dtype = "bfloat16" if device.type == "cuda" else "float32"
    return PI05Config(
        device=str(device),
        dtype=dtype,
        num_inference_steps=args.num_inference_steps,
        num_action_samples=args.num_action_samples,
        compile_model=args.compile_model,
        input_features={
            "observation.images.front": PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, args.image_height, args.image_width),
            ),
            "observation.state": PolicyFeature(
                type=FeatureType.STATE,
                shape=(args.state_dim,),
            ),
        },
        output_features={
            "action": PolicyFeature(
                type=FeatureType.ACTION,
                shape=(args.action_dim,),
            )
        },
        normalization_mapping={
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.IDENTITY,
        },
    )


def make_dummy_observation(args: argparse.Namespace) -> dict[str, np.ndarray]:
    return {
        "observation.images.front": np.zeros((args.image_height, args.image_width, 3), dtype=np.uint8),
        "observation.state": np.zeros((args.state_dim,), dtype=np.float32),
    }


def load_raw_observation(args: argparse.Namespace) -> dict[str, np.ndarray]:
    if args.observation_json:
        observation, _metadata = build_pi05_observation_from_robot_observer_json(args.observation_json)
        return observation
    return make_dummy_observation(args)


def clone_raw_observation(raw_observation: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {key: value.copy() for key, value in raw_observation.items()}


@contextmanager
def maybe_record_function(enabled: bool, label: str):
    if enabled and hasattr(torch, "profiler"):
        with torch.profiler.record_function(label):
            yield
    else:
        yield


def stage_ms(t0: float, t1: float) -> float:
    return (t1 - t0) * 1000.0


def parse_benchmark_steps(raw_steps: str | None) -> list[int]:
    if raw_steps is None:
        return []
    steps = []
    for part in raw_steps.split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value <= 0:
            raise ValueError(f"benchmark step count must be positive, got {value}")
        steps.append(value)
    return steps


def set_num_inference_steps(policy: PI05Policy, num_inference_steps: int) -> None:
    policy.config.num_inference_steps = num_inference_steps
    if hasattr(policy, "model") and hasattr(policy.model, "config"):
        policy.model.config.num_inference_steps = num_inference_steps


def summarize_timings(run_timings: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    if not run_timings:
        return summary
    for key in run_timings[0]:
        values = [timings[key] for timings in run_timings]
        summary[key] = {
            "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
        }
    return summary


def run_single_inference(
    *,
    raw_observation: dict[str, np.ndarray],
    task: str,
    runtime_device: torch.device,
    preprocess,
    postprocess,
    policy: PI05Policy,
    actions_per_chunk: int,
    num_action_samples: int,
    record_functions: bool,
) -> tuple[dict[str, float], np.ndarray]:
    requested_actions = max(1, min(actions_per_chunk, policy.config.chunk_size))
    timings: dict[str, float] = {}

    synchronize_if_needed(runtime_device)
    total_t0 = time.perf_counter()

    with maybe_record_function(record_functions, "pi05.profile.prepare_observation"):
        stage_t0 = time.perf_counter()
        obs = prepare_observation_for_inference(clone_raw_observation(raw_observation), runtime_device, task=task)
        synchronize_if_needed(runtime_device)
        stage_t1 = time.perf_counter()
        timings["prepare_observation_ms"] = stage_ms(stage_t0, stage_t1)

    with maybe_record_function(record_functions, "pi05.profile.preprocess"):
        stage_t0 = time.perf_counter()
        processed_obs = preprocess(obs)
        synchronize_if_needed(runtime_device)
        stage_t1 = time.perf_counter()
        timings["preprocess_ms"] = stage_ms(stage_t0, stage_t1)

    with maybe_record_function(record_functions, "pi05.profile.predict_action_chunk"):
        stage_t0 = time.perf_counter()
        action_chunk = policy.predict_action_chunk(
            processed_obs,
            num_action_samples=num_action_samples,
        )
        if action_chunk.ndim == 4:
            action_chunk = action_chunk[:, :, :requested_actions, :]
        else:
            action_chunk = action_chunk[:, :requested_actions, :]
        synchronize_if_needed(runtime_device)
        stage_t1 = time.perf_counter()
        timings["predict_action_chunk_ms"] = stage_ms(stage_t0, stage_t1)

    with maybe_record_function(record_functions, "pi05.profile.postprocess"):
        stage_t0 = time.perf_counter()
        if action_chunk.ndim == 4:
            processed_samples = []
            for sample_idx in range(action_chunk.shape[1]):
                processed_actions = []
                for idx in range(action_chunk.shape[2]):
                    processed_action = postprocess(action_chunk[:, sample_idx, idx, :])
                    processed_actions.append(processed_action)
                processed_samples.append(torch.stack(processed_actions, dim=1))
            action_tensor = torch.stack(processed_samples, dim=1).squeeze(0)
        else:
            processed_actions = []
            for idx in range(action_chunk.shape[1]):
                processed_action = postprocess(action_chunk[:, idx, :])
                processed_actions.append(processed_action)
            action_tensor = torch.stack(processed_actions, dim=1).squeeze(0)
        synchronize_if_needed(runtime_device)
        stage_t1 = time.perf_counter()
        timings["postprocess_ms"] = stage_ms(stage_t0, stage_t1)

    with maybe_record_function(record_functions, "pi05.profile.to_numpy"):
        stage_t0 = time.perf_counter()
        action_np = action_tensor.detach().cpu().numpy()
        stage_t1 = time.perf_counter()
        timings["to_numpy_ms"] = stage_ms(stage_t0, stage_t1)

    synchronize_if_needed(runtime_device)
    total_t1 = time.perf_counter()
    timings["total_ms"] = stage_ms(total_t0, total_t1)
    return timings, action_np


def print_summary(name: str, values: list[float]) -> None:
    if not values:
        return
    mean_value = statistics.mean(values)
    std_value = statistics.stdev(values) if len(values) > 1 else 0.0
    print(
        f"{name}: mean={mean_value:.2f} ms std={std_value:.2f} ms "
        f"min={min(values):.2f} ms max={max(values):.2f} ms"
    )


def run_benchmark_mode(
    *,
    benchmark_steps: list[int],
    raw_observation: dict[str, np.ndarray],
    task: str,
    runtime_device: torch.device,
    preprocess,
    postprocess,
    policy: PI05Policy,
    actions_per_chunk: int,
    num_action_samples: int,
    warmup_runs: int,
    profile_runs: int,
) -> None:
    print("benchmark_mode: num_inference_steps sweep")
    print("benchmark_steps:", benchmark_steps)

    benchmark_rows: list[dict[str, float]] = []
    for num_steps in benchmark_steps:
        set_num_inference_steps(policy, num_steps)
        for _ in range(max(0, warmup_runs)):
            run_single_inference(
                raw_observation=raw_observation,
                task=task,
                runtime_device=runtime_device,
                preprocess=preprocess,
                postprocess=postprocess,
                policy=policy,
                actions_per_chunk=actions_per_chunk,
                num_action_samples=num_action_samples,
                record_functions=False,
            )

        run_timings: list[dict[str, float]] = []
        for _ in range(max(1, profile_runs)):
            timings, _ = run_single_inference(
                raw_observation=raw_observation,
                task=task,
                runtime_device=runtime_device,
                preprocess=preprocess,
                postprocess=postprocess,
                policy=policy,
                actions_per_chunk=actions_per_chunk,
                num_action_samples=num_action_samples,
                record_functions=False,
            )
            run_timings.append(timings)

        summary = summarize_timings(run_timings)
        benchmark_rows.append(
            {
                "num_inference_steps": float(num_steps),
                "prepare_observation_ms": summary["prepare_observation_ms"]["mean"],
                "preprocess_ms": summary["preprocess_ms"]["mean"],
                "predict_action_chunk_ms": summary["predict_action_chunk_ms"]["mean"],
                "postprocess_ms": summary["postprocess_ms"]["mean"],
                "to_numpy_ms": summary["to_numpy_ms"]["mean"],
                "total_ms": summary["total_ms"]["mean"],
                "total_ms_std": summary["total_ms"]["std"],
            }
        )

        print(
            f"benchmark_step_result: steps={num_steps} "
            f"predict_action_chunk_ms={summary['predict_action_chunk_ms']['mean']:.2f} "
            f"total_ms={summary['total_ms']['mean']:.2f} "
            f"total_ms_std={summary['total_ms']['std']:.2f}"
        )

    baseline_steps = max(int(row["num_inference_steps"]) for row in benchmark_rows)
    baseline_total_ms = next(
        row["total_ms"] for row in benchmark_rows if int(row["num_inference_steps"]) == baseline_steps
    )

    print("benchmark_summary_csv:")
    print(
        "num_inference_steps,prepare_observation_ms,preprocess_ms,predict_action_chunk_ms,"
        "postprocess_ms,to_numpy_ms,total_ms,total_ms_std,speedup_vs_max_steps"
    )
    for row in benchmark_rows:
        speedup = baseline_total_ms / row["total_ms"] if row["total_ms"] > 0 else 0.0
        print(
            f"{int(row['num_inference_steps'])},"
            f"{row['prepare_observation_ms']:.2f},"
            f"{row['preprocess_ms']:.2f},"
            f"{row['predict_action_chunk_ms']:.2f},"
            f"{row['postprocess_ms']:.2f},"
            f"{row['to_numpy_ms']:.2f},"
            f"{row['total_ms']:.2f},"
            f"{row['total_ms_std']:.2f},"
            f"{speedup:.3f}"
        )


def main() -> None:
    args = parse_args()
    benchmark_steps = parse_benchmark_steps(args.benchmark_steps)
    if benchmark_steps and args.torch_profiler:
        raise ValueError("--benchmark-steps cannot be combined with --torch-profiler")

    runtime_device = select_device(args.device)
    config = build_config(args, runtime_device)
    load_config = copy.deepcopy(config)
    load_config.device = "cpu"
    tokenizer_path = resolve_tokenizer_path(args.model_id, args.tokenizer_path)

    print(f"loading_policy_from: {args.model_id}")
    print(f"runtime_device: {runtime_device}")
    print(f"num_inference_steps: {args.num_inference_steps}")
    print(f"num_action_samples: {args.num_action_samples}")
    print(f"compile_model: {args.compile_model}")
    print(f"actions_per_chunk: {args.actions_per_chunk}")
    print(f"warmup_runs: {args.warmup_runs}")
    print(f"profile_runs: {args.profile_runs}")
    if benchmark_steps:
        print(f"benchmark_steps: {benchmark_steps}")

    policy = PI05Policy.from_pretrained(args.model_id, config=load_config)
    if str(runtime_device) != "cpu":
        policy.model.to(runtime_device)
    policy.config.device = str(runtime_device)
    preprocess, postprocess = make_pi05_pre_post_processors(
        policy.config,
        tokenizer_name_or_path=tokenizer_path,
    )
    raw_observation = load_raw_observation(args)

    if benchmark_steps:
        run_benchmark_mode(
            benchmark_steps=benchmark_steps,
            raw_observation=raw_observation,
            task=args.task,
            runtime_device=runtime_device,
            preprocess=preprocess,
            postprocess=postprocess,
            policy=policy,
            actions_per_chunk=args.actions_per_chunk,
            num_action_samples=args.num_action_samples,
            warmup_runs=args.warmup_runs,
            profile_runs=args.profile_runs,
        )
        return

    for _ in range(max(0, args.warmup_runs)):
        run_single_inference(
            raw_observation=raw_observation,
            task=args.task,
            runtime_device=runtime_device,
            preprocess=preprocess,
            postprocess=postprocess,
            policy=policy,
            actions_per_chunk=args.actions_per_chunk,
            num_action_samples=args.num_action_samples,
            record_functions=False,
        )

    run_timings: list[dict[str, float]] = []
    final_action = None

    if args.torch_profiler:
        if not hasattr(torch, "profiler"):
            raise RuntimeError("This torch build does not provide torch.profiler")

        activities = [torch.profiler.ProfilerActivity.CPU]
        if runtime_device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)

        trace_dir = Path(args.trace_dir)
        trace_dir.mkdir(parents=True, exist_ok=True)
        sort_by = args.sort_by or (
            "self_cuda_time_total" if runtime_device.type == "cuda" else "self_cpu_time_total"
        )

        with torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(wait=0, warmup=1, active=max(1, args.profile_runs), repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(str(trace_dir)),
            record_shapes=args.record_shapes,
            profile_memory=args.profile_memory,
            with_stack=False,
            acc_events=True,
        ) as prof:
            for step_idx in range(1 + max(1, args.profile_runs)):
                timings, action_np = run_single_inference(
                    raw_observation=raw_observation,
                    task=args.task,
                    runtime_device=runtime_device,
                    preprocess=preprocess,
                    postprocess=postprocess,
                    policy=policy,
                    actions_per_chunk=args.actions_per_chunk,
                    num_action_samples=args.num_action_samples,
                    record_functions=True,
                )
                if step_idx >= 1:
                    run_timings.append(timings)
                    final_action = action_np
                prof.step()

        print(f"torch_profiler_trace_dir: {trace_dir}")
        print("torch_profiler_table:")
        print(prof.key_averages().table(sort_by=sort_by, row_limit=args.row_limit))
    else:
        for _ in range(max(1, args.profile_runs)):
            timings, action_np = run_single_inference(
                raw_observation=raw_observation,
                task=args.task,
                runtime_device=runtime_device,
                preprocess=preprocess,
                postprocess=postprocess,
                policy=policy,
                actions_per_chunk=args.actions_per_chunk,
                num_action_samples=args.num_action_samples,
                record_functions=False,
            )
            run_timings.append(timings)
            final_action = action_np

    print("timing_summary:")
    for key in [
        "prepare_observation_ms",
        "preprocess_ms",
        "predict_action_chunk_ms",
        "postprocess_ms",
        "to_numpy_ms",
        "total_ms",
    ]:
        print_summary(key, [timings[key] for timings in run_timings])

    if final_action is not None:
        print("last_action_chunk_shape:", tuple(final_action.shape))
        if final_action.size > 0:
            if final_action.ndim == 3:
                print("last_first_action:", final_action[0, 0].tolist())
            else:
                print("last_first_action:", final_action[0].tolist())


if __name__ == "__main__":
    main()

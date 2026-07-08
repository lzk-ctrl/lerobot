#!/usr/bin/env python

from __future__ import annotations

import argparse
import csv
import json
import socket
import sys
import time
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from pi05_runtime import (  # noqa: E402
    build_pi05_observation_from_robot_observer_json,
    make_dummy_observation,
    recv_message,
    send_message,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run repeated PI0.5 server inference on the same observation and compare action variance."
    )
    parser.add_argument("--server-host", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=6001)
    parser.add_argument("--task", default="walk forward while avoiding all obstacles")
    parser.add_argument("--observation-json", default=None)
    parser.add_argument("--runs", type=int, default=1, help="Number of repeated inference requests.")
    parser.add_argument("--actions-per-chunk", type=int, default=20)
    parser.add_argument(
        "--num-action-samples",
        type=int,
        default=None,
        help="Number of action samples to request per inference. Omit to use the server default.",
    )
    parser.add_argument("--cmd-dims", type=int, default=3, help="Number of leading action dims to summarize.")
    parser.add_argument("--socket-timeout-sec", type=float, default=60.0)
    parser.add_argument("--sleep-sec", type=float, default=0.0, help="Optional pause between requests.")
    parser.add_argument(
        "--initial-noise-path",
        default=None,
        help="Optional fixed initial noise. Omit this to measure random-noise variability.",
    )
    parser.add_argument(
        "--save-final-noise-path",
        default=None,
        help="Optional path passed to the server for saving each run's final denoised tensor.",
    )
    parser.add_argument(
        "--csv-path",
        default=None,
        help="Optional CSV path for per-run first-action results.",
    )
    return parser.parse_args()


def load_observation(args: argparse.Namespace) -> dict[str, np.ndarray]:
    if args.observation_json:
        observation, _metadata = build_pi05_observation_from_robot_observer_json(args.observation_json)
        return observation
    return make_dummy_observation()


def request_action(sock: socket.socket, args: argparse.Namespace, observation: dict[str, np.ndarray]) -> dict:
    payload = {
        "type": "infer",
        "task": args.task,
        "observation": observation,
        "actions_per_chunk": args.actions_per_chunk,
    }
    if args.num_action_samples is not None:
        payload["num_action_samples"] = args.num_action_samples
    if args.initial_noise_path:
        payload["initial_noise_path"] = str(Path(args.initial_noise_path).expanduser())
    if args.save_final_noise_path:
        payload["save_final_noise_path"] = str(Path(args.save_final_noise_path).expanduser())

    t0 = time.perf_counter()
    send_message(sock, payload)
    response = recv_message(sock)
    response["roundtrip_time_ms"] = (time.perf_counter() - t0) * 1000.0
    return response


def pairwise_l2(values: np.ndarray) -> np.ndarray:
    distances = []
    for i in range(values.shape[0]):
        for j in range(i + 1, values.shape[0]):
            distances.append(float(np.linalg.norm(values[i] - values[j], ord=2)))
    return np.asarray(distances, dtype=np.float32)


def print_vector(name: str, value: np.ndarray) -> None:
    print(f"{name}: {np.asarray(value, dtype=np.float32).tolist()}")


def write_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_action_samples_chunk_csv(path: Path, action_samples: np.ndarray) -> None:
    """Write one request's samples as rows and chunk timesteps as columns."""
    path.parent.mkdir(parents=True, exist_ok=True)
    num_actions = int(action_samples.shape[1])
    fieldnames = [f"action_{idx:02d}" for idx in range(num_actions)]
    rows = []
    for sample_idx in range(action_samples.shape[0]):
        row = {
            f"action_{action_idx:02d}": json.dumps(
                action_samples[sample_idx, action_idx].astype(float).tolist(),
                separators=(",", ":"),
            )
            for action_idx in range(num_actions)
        }
        rows.append(row)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.runs < 1:
        raise ValueError("--runs must be at least 1")

    observation = load_observation(args)
    action_chunks = []
    first_actions = []
    sample_first_actions = []
    within_run_rows: list[dict[str, float | int]] = []
    rows: list[dict[str, float | int]] = []

    print(f"server: {args.server_host}:{args.server_port}")
    print(f"runs: {args.runs}")
    print(f"actions_per_chunk: {args.actions_per_chunk}")
    print(f"num_action_samples: {args.num_action_samples or 'server_default'}")
    print(f"initial_noise_path: {args.initial_noise_path or 'random_each_run'}")

    with socket.create_connection((args.server_host, args.server_port), timeout=args.socket_timeout_sec) as sock:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        for run_idx in range(args.runs):
            response = request_action(sock, args, observation)
            if not response.get("ok", False):
                raise RuntimeError(response.get("error", "Unknown PI05 server error"))

            raw_action_chunk = np.asarray(response["action_chunk"], dtype=np.float32)
            if raw_action_chunk.ndim == 2:
                action_samples = raw_action_chunk[None, ...]
            elif raw_action_chunk.ndim == 3:
                action_samples = raw_action_chunk
            else:
                raise RuntimeError(f"Malformed action_chunk shape: {raw_action_chunk.shape}")

            if action_samples.shape[-1] < args.cmd_dims:
                raise RuntimeError(f"Malformed action_chunk shape: {raw_action_chunk.shape}")

            action_chunk = action_samples[0]
            first_action = action_samples[0, 0]
            sample_first_cmds = action_samples[:, 0, : args.cmd_dims]
            sample_first_actions.append(action_samples[:, 0, :])
            action_chunks.append(action_samples.reshape(action_samples.shape[0], -1))
            first_actions.append(first_action)

            cmd = first_action[: args.cmd_dims]
            if action_samples.shape[0] > 1:
                sample_pairwise_cmd = pairwise_l2(sample_first_cmds)
                sample_pairwise_chunk = pairwise_l2(action_samples.reshape(action_samples.shape[0], -1))
                sample_l2_mean = float(sample_pairwise_cmd.mean())
                sample_l2_max = float(sample_pairwise_cmd.max())
                chunk_l2_mean = float(sample_pairwise_chunk.mean())
                chunk_l2_max = float(sample_pairwise_chunk.max())
            else:
                sample_l2_mean = 0.0
                sample_l2_max = 0.0
                chunk_l2_mean = 0.0
                chunk_l2_max = 0.0

            print(
                f"run_{run_idx:03d}: "
                f"inference_ms={float(response['inference_time_ms']):.2f} "
                f"roundtrip_ms={float(response['roundtrip_time_ms']):.2f} "
                f"samples={action_samples.shape[0]} "
                f"selected_first_cmd={cmd.tolist()} "
                f"sample_pairwise_l2_mean={sample_l2_mean:.6f} "
                f"sample_pairwise_l2_max={sample_l2_max:.6f}"
            )
            for sample_idx, sample_cmd in enumerate(sample_first_cmds):
                print(f"  sample_{sample_idx:02d}_first_cmd={sample_cmd.tolist()}")

            if args.runs == 1:
                if args.csv_path:
                    csv_path = Path(args.csv_path).expanduser()
                    write_action_samples_chunk_csv(csv_path, action_samples)
                    print(f"chunk_samples_csv_path: {csv_path}")
                    print(
                        "chunk_samples_csv_shape: "
                        f"rows={action_samples.shape[0]} cols={action_samples.shape[1]}"
                    )
                return

            row: dict[str, float | int] = {
                "run": run_idx,
                "inference_time_ms": float(response["inference_time_ms"]),
                "roundtrip_time_ms": float(response["roundtrip_time_ms"]),
                "num_action_samples": int(action_samples.shape[0]),
                "sample_first_cmd_pairwise_l2_mean": sample_l2_mean,
                "sample_first_cmd_pairwise_l2_max": sample_l2_max,
                "sample_action_chunk_pairwise_l2_mean": chunk_l2_mean,
                "sample_action_chunk_pairwise_l2_max": chunk_l2_max,
            }
            for dim_idx, value in enumerate(cmd):
                row[f"selected_cmd_{dim_idx}"] = float(value)
            rows.append(row)

            for sample_idx, sample_cmd in enumerate(sample_first_cmds):
                sample_row: dict[str, float | int] = {
                    "run": run_idx,
                    "sample": sample_idx,
                    "inference_time_ms": float(response["inference_time_ms"]),
                    "roundtrip_time_ms": float(response["roundtrip_time_ms"]),
                }
                for dim_idx, value in enumerate(sample_cmd):
                    sample_row[f"cmd_{dim_idx}"] = float(value)
                within_run_rows.append(sample_row)

            if args.sleep_sec > 0:
                time.sleep(args.sleep_sec)

    first_actions_arr = np.stack(first_actions, axis=0)
    cmd_arr = first_actions_arr[:, : args.cmd_dims]
    flat_chunks = np.stack([chunk[0] for chunk in action_chunks], axis=0)
    pairwise_cmd = pairwise_l2(cmd_arr)
    pairwise_chunk = pairwise_l2(flat_chunks)
    sample_first_actions_arr = np.concatenate(sample_first_actions, axis=0)
    sample_cmd_arr = sample_first_actions_arr[:, : args.cmd_dims]
    all_sample_pairwise_cmd = pairwise_l2(sample_cmd_arr)

    print("summary:")
    print_vector("selected_first_cmd_mean", cmd_arr.mean(axis=0))
    print_vector("selected_first_cmd_std", cmd_arr.std(axis=0))
    print_vector("selected_first_cmd_min", cmd_arr.min(axis=0))
    print_vector("selected_first_cmd_max", cmd_arr.max(axis=0))
    print(f"selected_first_cmd_l2_std_norm: {float(np.linalg.norm(cmd_arr.std(axis=0), ord=2)):.6f}")
    print(f"selected_first_cmd_pairwise_l2_mean: {float(pairwise_cmd.mean()):.6f}")
    print(f"selected_first_cmd_pairwise_l2_max: {float(pairwise_cmd.max()):.6f}")
    print(f"selected_action_chunk_pairwise_l2_mean: {float(pairwise_chunk.mean()):.6f}")
    print(f"selected_action_chunk_pairwise_l2_max: {float(pairwise_chunk.max()):.6f}")
    print_vector("all_samples_first_cmd_mean", sample_cmd_arr.mean(axis=0))
    print_vector("all_samples_first_cmd_std", sample_cmd_arr.std(axis=0))
    print_vector("all_samples_first_cmd_min", sample_cmd_arr.min(axis=0))
    print_vector("all_samples_first_cmd_max", sample_cmd_arr.max(axis=0))
    print(f"all_samples_first_cmd_l2_std_norm: {float(np.linalg.norm(sample_cmd_arr.std(axis=0), ord=2)):.6f}")
    print(f"all_samples_first_cmd_pairwise_l2_mean: {float(all_sample_pairwise_cmd.mean()):.6f}")
    print(f"all_samples_first_cmd_pairwise_l2_max: {float(all_sample_pairwise_cmd.max()):.6f}")
    if within_run_rows:
        within_means = np.asarray(
            [row["sample_first_cmd_pairwise_l2_mean"] for row in rows], dtype=np.float32
        )
        within_maxes = np.asarray(
            [row["sample_first_cmd_pairwise_l2_max"] for row in rows], dtype=np.float32
        )
        print(f"within_request_sample_pairwise_l2_mean_avg: {float(within_means.mean()):.6f}")
        print(f"within_request_sample_pairwise_l2_max: {float(within_maxes.max()):.6f}")

    if args.csv_path:
        csv_path = Path(args.csv_path).expanduser()
        write_csv(csv_path, rows)
        print(f"csv_path: {csv_path}")
        sample_csv_path = csv_path.with_name(f"{csv_path.stem}_samples{csv_path.suffix}")
        write_csv(sample_csv_path, within_run_rows)
        print(f"sample_csv_path: {sample_csv_path}")


if __name__ == "__main__":
    main()

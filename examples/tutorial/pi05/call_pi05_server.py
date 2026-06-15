#!/usr/bin/env python

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shlex
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
REPO_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_ROS_HOME = REPO_ROOT / ".ros_home"
DEFAULT_ROS_LOG_DIR = REPO_ROOT / ".ros_logs"
DEFAULT_FIRST_OBSERVATION_HTML = REPO_ROOT / "tmp" / "first_observation.html"
DEFAULT_METRICS_CSV = REPO_ROOT / "tmp" / "call_pi05_cycle_metrics.csv"

from pi05_runtime import (  # noqa: E402
    build_pi05_observation_from_robot_observer_json,
    build_pi05_observation_from_robot_observer_snapshot,
    build_cmd_vel,
    make_dummy_observation,
    recv_message,
    save_observation_debug_html,
    send_message,
    send_udp_cmd_vel,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Call a persistent PI0.5 inference server")
    parser.add_argument("--server-host", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=6001)
    parser.add_argument("--task", default="walk forward while avoiding all obstacles")
    parser.add_argument("--state-dim", type=int, default=22)
    parser.add_argument("--action-dim", type=int, default=3)
    parser.add_argument("--image-height", type=int, default=480)
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument(
        "--snapshot-server-url",
        default=None,
        help="HTTP endpoint for a persistent observation snapshot server, e.g. http://127.0.0.1:7001/latest.",
    )
    parser.add_argument(
        "--snapshot-request-timeout-sec",
        type=float,
        default=3.0,
        help="HTTP timeout for requesting the latest observation snapshot.",
    )
    parser.add_argument(
        "--snapshot-wait-timeout-sec",
        type=float,
        default=0.25,
        help="How long the snapshot server may wait for required messages before replying.",
    )
    parser.add_argument(
        "--legacy-robot-observer",
        action="store_true",
        help="Use the older per-cycle robot_observer.py subprocess mode.",
    )
    parser.add_argument(
        "--robot-observer",
        action="store_true",
        help="Alias for --legacy-robot-observer.",
    )
    parser.add_argument(
        "--robot-observer-python",
        default="/usr/bin/python3",
        help="Python interpreter used to run robot_observer.py.",
    )
    parser.add_argument(
        "--robot-observer-script",
        default="~/eai_ws/install/champ_teleop/lib/champ_teleop/robot_observer.py",
        help="Path to robot_observer.py.",
    )
    parser.add_argument(
        "--robot-observer-shell",
        default="/bin/bash",
        help="Shell used to source ROS setup files before running robot_observer.py.",
    )
    parser.add_argument(
        "--robot-observer-ros-setup",
        default="/opt/ros/humble/setup.bash",
        help="ROS setup script sourced before robot_observer.py.",
    )
    parser.add_argument(
        "--robot-observer-workspace-setup",
        default="~/eai_ws/install/setup.bash",
        help="Workspace setup script sourced before robot_observer.py.",
    )
    parser.add_argument(
        "--robot-observer-ros-home",
        default=str(DEFAULT_ROS_HOME),
        help="ROS_HOME used for the robot_observer.py subprocess.",
    )
    parser.add_argument(
        "--robot-observer-ros-log-dir",
        default=str(DEFAULT_ROS_LOG_DIR),
        help="ROS_LOG_DIR used for the robot_observer.py subprocess.",
    )
    parser.add_argument(
        "--observer-timeout-sec",
        type=float,
        default=0.3,
        help="Sampling time passed to robot_observer.py --timeout.",
    )
    parser.add_argument(
        "--observer-subprocess-timeout-sec",
        type=float,
        default=5.0,
        help="Wall-clock timeout for the robot_observer.py subprocess.",
    )
    parser.add_argument(
        "--observation-json",
        default=None,
        help=(
            "Path to a JSON snapshot produced by robot_observer.py. "
            "Use --include-image-data when generating the snapshot."
        ),
    )
    parser.add_argument(
        "--observation-source",
        choices=["snapshot_server", "legacy_robot_observer", "json", "dummy"],
        default=None,
        help="Explicitly choose where observations come from.",
    )
    parser.add_argument(
        "--print-state-layout",
        action="store_true",
        help="Print the extracted observation.state feature names before sending the request.",
    )
    parser.add_argument(
        "--first-observation-html-path",
        default=str(DEFAULT_FIRST_OBSERVATION_HTML),
        help="Where to save the first real observation snapshot as an HTML page.",
    )
    parser.add_argument(
        "--metrics-csv-path",
        default=str(DEFAULT_METRICS_CSV),
        help="Append per-cycle timing and similarity metrics to this CSV file.",
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Keep capturing observations, calling the VLA, and sending actions in a loop.",
    )
    parser.add_argument(
        "--execution-mode",
        choices=["continuous_control", "task_execution"],
        default="task_execution",
        help=(
            "continuous_control sends each inferred cmd_vel immediately and replans right away. "
            "task_execution holds one inferred cmd_vel for a short horizon, then sends zero velocity before replanning."
        ),
    )
    parser.add_argument(
        "--execution-horizon-sec",
        type=float,
        default=0.1,
        help="How long to hold one inferred cmd_vel in task_execution mode.",
    )
    parser.add_argument(
        "--execution-settle-sec",
        type=float,
        default=0.05,
        help="How long to wait after sending zero cmd_vel in task_execution mode before the next observation.",
    )
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=0,
        help="Maximum continuous cycles to run. 0 means run forever.",
    )
    parser.add_argument(
        "--safety-stop-distance-m",
        type=float,
        default=0.35,
        help="Hard stop threshold for front obstacle distance in meters.",
    )
    parser.add_argument(
        "--safety-slow-distance-m",
        type=float,
        default=0.80,
        help="Start scaling down forward speed below this front obstacle distance in meters.",
    )
    parser.add_argument(
        "--cmd-vel-output",
        choices=["print", "udp", "print+udp"],
        default="print",
        help="How to handle the first 3 action dimensions after inference.",
    )
    parser.add_argument("--udp-host", default="127.0.0.1", help="UDP bridge host")
    parser.add_argument("--udp-port", type=int, default=5555, help="UDP bridge port")
    parser.add_argument("--max-linear-x", type=float, default=0.3)
    parser.add_argument("--max-linear-y", type=float, default=0.25)
    parser.add_argument("--max-angular-z", type=float, default=0.5)
    parser.add_argument(
        "--actions-per-chunk",
        type=int,
        default=1,
        help="How many actions to request from the server.",
    )
    parser.add_argument(
        "--num-action-samples",
        type=int,
        default=None,
        help=(
            "How many noise samples/action chunk candidates to request from the server. "
            "Omit to use the server default."
        ),
    )
    parser.add_argument(
        "--action-sample-index",
        type=int,
        default=0,
        help="Which returned action sample candidate to execute when the server returns multiple samples.",
    )
    parser.add_argument(
        "--send-mode",
        choices=["first", "chunk"],
        default="first",
        help="Send only the first action or stream the whole returned chunk.",
    )
    parser.add_argument(
        "--control-rate-hz",
        type=float,
        default=20.0,
        help="Publishing rate used when --send-mode=chunk.",
    )
    parser.add_argument(
        "--socket-timeout-sec",
        type=float,
        default=60.0,
        help="Timeout for the TCP request to the inference server.",
    )
    parser.add_argument(
        "--initial-noise-path",
        default=None,
        help=(
            "Optional .pt path containing an initial noise tensor to use for this request. "
            "If omitted, the server samples random noise."
        ),
    )
    parser.add_argument(
        "--save-final-noise-path",
        default=None,
        help=(
            "Optional .pt path where the server should save this request's final denoised result "
            "for future warm-start reuse."
        ),
    )
    return parser.parse_args()


def request_action_chunk(args: argparse.Namespace, observation: dict[str, np.ndarray]) -> dict:
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

    with socket.create_connection((args.server_host, args.server_port), timeout=args.socket_timeout_sec) as sock:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        t0 = time.perf_counter()
        send_message(sock, payload)
        response = recv_message(sock)
        roundtrip_ms = (time.perf_counter() - t0) * 1000.0

    response["roundtrip_time_ms"] = roundtrip_ms
    return response


def capture_observation_via_snapshot_server(
    args: argparse.Namespace,
) -> tuple[dict[str, np.ndarray], dict]:
    if not args.snapshot_server_url:
        raise ValueError("--snapshot-server-url is required for snapshot server mode")

    query = urlencode(
        {
            "include_image_data": 1,
            "timeout": args.snapshot_wait_timeout_sec,
        }
    )
    separator = "&" if "?" in args.snapshot_server_url else "?"
    request_url = f"{args.snapshot_server_url}{separator}{query}"
    with urlopen(request_url, timeout=args.snapshot_request_timeout_sec) as response:
        payload = response.read().decode("utf-8")

    snapshot = json.loads(payload)
    return build_pi05_observation_from_robot_observer_snapshot(snapshot)


def capture_observation_via_robot_observer(
    args: argparse.Namespace,
) -> tuple[dict[str, np.ndarray], dict]:
    script_path = str(Path(args.robot_observer_script).expanduser())
    ros_setup = str(Path(args.robot_observer_ros_setup).expanduser())
    workspace_setup = str(Path(args.robot_observer_workspace_setup).expanduser())

    setup_commands: list[str] = []
    if ros_setup:
        setup_commands.append(f"source {shlex.quote(ros_setup)}")
    if workspace_setup:
        setup_commands.append(f"source {shlex.quote(workspace_setup)}")

    observer_command = " ".join(
        [
            shlex.quote(args.robot_observer_python),
            shlex.quote(script_path),
            "--timeout",
            shlex.quote(str(args.observer_timeout_sec)),
            "--include-image-data",
            "--compact",
        ]
    )
    shell_command = " && ".join([*setup_commands, observer_command])
    ros_home = Path(args.robot_observer_ros_home).expanduser()
    ros_log_dir = Path(args.robot_observer_ros_log_dir).expanduser()
    ros_home.mkdir(parents=True, exist_ok=True)
    ros_log_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["ROS_HOME"] = str(ros_home)
    env["ROS_LOG_DIR"] = str(ros_log_dir)

    try:
        result = subprocess.run(
            [args.robot_observer_shell, "-c", shell_command],
            check=True,
            capture_output=True,
            text=True,
            timeout=args.observer_subprocess_timeout_sec,
            env=env,
        )
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() if exc.stderr else ""
        stdout = exc.stdout.strip() if exc.stdout else ""
        details = stderr or stdout or f"exit code {exc.returncode}"
        raise RuntimeError(f"robot_observer.py failed: {details}") from exc

    stdout = result.stdout.strip()
    if not stdout:
        stderr = result.stderr.strip() if result.stderr else ""
        raise RuntimeError(f"robot_observer.py returned empty stdout. stderr={stderr!r}")

    json_start = stdout.find("{")
    json_end = stdout.rfind("}")
    if json_start < 0 or json_end < json_start:
        stderr = result.stderr.strip() if result.stderr else ""
        raise RuntimeError(
            "robot_observer.py did not return JSON. "
            f"stdout={stdout[:300]!r} stderr={stderr[:300]!r}"
        )

    snapshot = json.loads(stdout[json_start : json_end + 1])
    return build_pi05_observation_from_robot_observer_snapshot(snapshot)


def load_observation(
    args: argparse.Namespace,
) -> tuple[dict[str, np.ndarray], dict | None]:
    observation_source = args.observation_source
    if observation_source is None:
        if args.snapshot_server_url:
            observation_source = "snapshot_server"
        elif args.legacy_robot_observer or args.robot_observer:
            observation_source = "legacy_robot_observer"
        elif args.observation_json:
            observation_source = "json"
        else:
            observation_source = "dummy"

    if observation_source == "snapshot_server":
        return capture_observation_via_snapshot_server(args)

    if observation_source == "legacy_robot_observer":
        return capture_observation_via_robot_observer(args)

    if observation_source == "json":
        return build_pi05_observation_from_robot_observer_json(args.observation_json)

    observation = make_dummy_observation(
        image_height=args.image_height,
        image_width=args.image_width,
        state_dim=args.state_dim,
    )
    return observation, None


def maybe_print_observation_metadata(metadata: dict | None, args: argparse.Namespace) -> None:
    if metadata is None:
        return

    print(f"loaded_observation_image_shape: {metadata['image_shape']}")
    print(f"loaded_observation_state_dim: {metadata['state_dim']}")

    obstacle_distance = metadata.get("front_obstacle_distance_m")
    if obstacle_distance is not None:
        print(f"front_obstacle_distance_m: {obstacle_distance:.3f}")
    else:
        print("front_obstacle_distance_m: unavailable")

    if args.print_state_layout:
        print("loaded_observation_state_names:")
        for name in metadata["state_names"]:
            print(f"  - {name}")


def maybe_save_first_observation_html(
    metadata: dict | None,
    args: argparse.Namespace,
) -> None:
    if metadata is None or getattr(args, "_first_observation_html_saved", False):
        return

    output_path = save_observation_debug_html(
        output_path=args.first_observation_html_path,
        metadata=metadata,
        task=args.task,
    )
    args._first_observation_html_saved = True
    print(f"saved_first_observation_html: {output_path}")


def _downsample_grayscale_image(image: np.ndarray, target_height: int = 32, target_width: int = 32) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected HWC RGB image, received shape {image.shape}")

    row_indices = np.linspace(0, image.shape[0] - 1, num=target_height, dtype=np.int32)
    col_indices = np.linspace(0, image.shape[1] - 1, num=target_width, dtype=np.int32)
    sampled = image[row_indices][:, col_indices]
    return sampled.astype(np.float32).mean(axis=2)


def compute_image_similarity_metrics(
    current_image: np.ndarray,
    previous_image: np.ndarray,
) -> dict[str, float]:
    current_small = _downsample_grayscale_image(current_image)
    previous_small = _downsample_grayscale_image(previous_image)

    mean_abs_diff = float(np.mean(np.abs(current_small - previous_small)))
    l1_similarity = max(0.0, 1.0 - mean_abs_diff / 255.0)

    current_vec = current_small.reshape(-1)
    previous_vec = previous_small.reshape(-1)
    current_centered = current_vec - float(current_vec.mean())
    previous_centered = previous_vec - float(previous_vec.mean())
    denom = float(np.linalg.norm(current_centered) * np.linalg.norm(previous_centered))
    correlation = 0.0 if denom <= 1e-8 else float(np.dot(current_centered, previous_centered) / denom)

    return {
        "image_similarity_l1": l1_similarity,
        "image_similarity_corr": correlation,
        "image_mean_abs_diff": mean_abs_diff,
    }


def compute_cmd_similarity_metrics(
    current_cmd: dict[str, float],
    previous_cmd: dict[str, float],
) -> dict[str, float]:
    current = np.asarray(
        [current_cmd["linear.x"], current_cmd["linear.y"], current_cmd["angular.z"]],
        dtype=np.float32,
    )
    previous = np.asarray(
        [previous_cmd["linear.x"], previous_cmd["linear.y"], previous_cmd["angular.z"]],
        dtype=np.float32,
    )

    delta_l2 = float(np.linalg.norm(current - previous))
    denom = float(np.linalg.norm(current) * np.linalg.norm(previous))
    cosine = 0.0 if denom <= 1e-8 else float(np.dot(current, previous) / denom)
    relative_similarity = 1.0 - delta_l2 / max(float(np.linalg.norm(current) + np.linalg.norm(previous)), 1e-6)

    return {
        "cmd_similarity_cosine": cosine,
        "cmd_delta_l2": delta_l2,
        "cmd_similarity_relative": relative_similarity,
    }


def maybe_print_observation_similarity(
    observation: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> dict[str, float] | None:
    current_image = observation.get("observation.images.front")
    if current_image is None:
        return None

    previous_image = getattr(args, "_previous_observation_image", None)
    if previous_image is None:
        print("image_similarity_to_prev: n/a (first cycle)")
        metrics = None
    else:
        metrics = compute_image_similarity_metrics(current_image, previous_image)
        print(
            "image_similarity_to_prev:",
            (
                f"l1={metrics['image_similarity_l1']:.4f} "
                f"corr={metrics['image_similarity_corr']:.4f} "
                f"mean_abs_diff={metrics['image_mean_abs_diff']:.2f}"
            ),
        )

    args._previous_observation_image = np.asarray(current_image, dtype=np.uint8).copy()
    return metrics


def maybe_print_cmd_similarity(
    current_cmd: dict[str, float],
    args: argparse.Namespace,
) -> dict[str, float] | None:
    previous_cmd = getattr(args, "_previous_cmd_vel", None)
    if previous_cmd is None:
        print("cmd_similarity_to_prev: n/a (first cycle)")
        metrics = None
    else:
        metrics = compute_cmd_similarity_metrics(current_cmd, previous_cmd)
        print(
            "cmd_similarity_to_prev:",
            (
                f"cosine={metrics['cmd_similarity_cosine']:.4f} "
                f"relative={metrics['cmd_similarity_relative']:.4f} "
                f"delta_l2={metrics['cmd_delta_l2']:.4f}"
            ),
        )

    args._previous_cmd_vel = dict(current_cmd)
    return metrics


def append_cycle_metrics_csv(args: argparse.Namespace, row: dict[str, object]) -> None:
    csv_path = Path(args.metrics_csv_path).expanduser()
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "wall_time_epoch_s",
        "cycle_index",
        "observation_source",
        "task",
        "image_height",
        "image_width",
        "state_dim",
        "front_obstacle_distance_m",
        "observation_load_time_ms",
        "server_roundtrip_time_ms",
        "server_inference_time_ms",
        "transport_and_rpc_overhead_ms",
        "cmd_send_time_ms",
        "cycle_total_time_ms",
        "execution_mode",
        "execution_horizon_sec",
        "execution_settle_sec",
        "received_actions",
        "first_raw_action_linear_x",
        "first_raw_action_linear_y",
        "first_raw_action_angular_z",
        "final_cmd_linear_x",
        "final_cmd_linear_y",
        "final_cmd_angular_z",
        "image_similarity_l1",
        "image_similarity_corr",
        "image_mean_abs_diff",
        "cmd_similarity_cosine",
        "cmd_similarity_relative",
        "cmd_delta_l2",
    ]

    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({name: row.get(name) for name in fieldnames})

    if not getattr(args, "_metrics_csv_announced", False):
        print(f"metrics_csv_path: {csv_path}")
        args._metrics_csv_announced = True


def apply_front_obstacle_safety(
    cmd_vel: dict[str, float],
    observation_metadata: dict | None,
    args: argparse.Namespace,
) -> tuple[dict[str, float], str | None]:
    if observation_metadata is None:
        return cmd_vel, None

    obstacle_distance = observation_metadata.get("front_obstacle_distance_m")
    if obstacle_distance is None or not math.isfinite(obstacle_distance):
        return cmd_vel, None

    safe_cmd = dict(cmd_vel)
    if obstacle_distance <= args.safety_stop_distance_m:
        if safe_cmd["linear.x"] > 0.0:
            safe_cmd["linear.x"] = 0.0
        return safe_cmd, f"front obstacle {obstacle_distance:.3f}m <= stop threshold"

    if obstacle_distance <= args.safety_slow_distance_m and safe_cmd["linear.x"] > 0.0:
        scale = (obstacle_distance - args.safety_stop_distance_m) / (
            args.safety_slow_distance_m - args.safety_stop_distance_m
        )
        scale = max(0.0, min(1.0, scale))
        safe_cmd["linear.x"] *= scale
        return safe_cmd, f"front obstacle {obstacle_distance:.3f}m -> scaled forward speed by {scale:.2f}"

    return safe_cmd, None


def handle_action(
    action: np.ndarray,
    args: argparse.Namespace,
    observation_metadata: dict | None = None,
    label: str = "cmd_vel",
) -> tuple[float, dict[str, float]]:
    t0 = time.perf_counter()
    cmd_vel = build_cmd_vel(
        action,
        max_linear_x=args.max_linear_x,
        max_linear_y=args.max_linear_y,
        max_angular_z=args.max_angular_z,
    )
    cmd_vel, safety_message = apply_front_obstacle_safety(cmd_vel, observation_metadata, args)
    if args.cmd_vel_output in {"print", "print+udp"}:
        print(f"{label}:", cmd_vel)
        if safety_message is not None:
            print("safety:", safety_message)
    if args.cmd_vel_output in {"udp", "print+udp"}:
        send_udp_cmd_vel(cmd_vel, udp_host=args.udp_host, udp_port=args.udp_port, source="pi05_server_client")
        print(f"sent_udp_cmd_vel: {args.udp_host}:{args.udp_port}")
    return (time.perf_counter() - t0) * 1000.0, cmd_vel


def execute_cycle(args: argparse.Namespace) -> None:
    cycle_t0 = time.perf_counter()

    observation_t0 = time.perf_counter()
    observation, metadata = load_observation(args)
    observation_dt_ms = (time.perf_counter() - observation_t0) * 1000.0

    maybe_print_observation_metadata(metadata, args)
    image_similarity_metrics = maybe_print_observation_similarity(observation, args)
    maybe_save_first_observation_html(metadata, args)

    response = request_action_chunk(args, observation)
    if not response.get("ok", False):
        raise RuntimeError(response.get("error", "Unknown PI05 server error"))

    action_chunk = np.asarray(response["action_chunk"], dtype=np.float32)
    if action_chunk.ndim == 3:
        print(f"received_action_samples: {action_chunk.shape[0]}")
        for sample_idx in range(action_chunk.shape[0]):
            print(f"sample_{sample_idx}_first_raw_action:", action_chunk[sample_idx, 0].tolist())
        if not 0 <= args.action_sample_index < action_chunk.shape[0]:
            raise RuntimeError(
                f"--action-sample-index {args.action_sample_index} is out of range for "
                f"{action_chunk.shape[0]} returned samples"
            )
        print(f"selected_action_sample_index: {args.action_sample_index}")
        action_chunk = action_chunk[args.action_sample_index]
    if action_chunk.ndim != 2 or action_chunk.shape[1] < 3:
        raise RuntimeError(f"Server returned malformed action_chunk shape: {action_chunk.shape}")

    print(f"server_roundtrip_time_ms: {response['roundtrip_time_ms']:.2f}")
    print(f"server_inference_time_ms: {response['inference_time_ms']:.2f}")
    transport_overhead_ms = response["roundtrip_time_ms"] - response["inference_time_ms"]
    print(f"transport_and_rpc_overhead_ms: {transport_overhead_ms:.2f}")
    print(f"observation_load_time_ms: {observation_dt_ms:.2f}")
    if "noise_source" in response:
        print(f"noise_source: {response['noise_source']}")
    if "noise_input_path" in response:
        print(f"noise_input_path: {response['noise_input_path']}")
    if "saved_noise_path" in response:
        print(f"saved_noise_path: {response['saved_noise_path']}")
    warm_start_cmd_metrics = response.get("warm_start_cmd_metrics")
    if warm_start_cmd_metrics:
        print("initial_cmd:", warm_start_cmd_metrics["initial_cmd"])
        print("final_cmd:", warm_start_cmd_metrics["final_cmd"])
        print("cmd_delta_xyz:", warm_start_cmd_metrics["cmd_delta_xyz"])
        print(f"cmd_delta_l2: {float(warm_start_cmd_metrics['cmd_delta_l2']):.6f}")
    denoising_debug = response.get("denoising_debug")
    if denoising_debug:
        print(f"denoising_debug_artifact: {denoising_debug['artifact_path']}")
        print(f"denoising_debug_summary: {denoising_debug['summary_path']}")
        for step_summary in denoising_debug.get("step_summaries", []):
            action_delta = step_summary["action_delta"]
            print(
                "denoise_step_"
                f"{int(step_summary['step_idx']):02d}: "
                f"time={float(step_summary['time']):.4f} "
                f"delta_l1_mean={float(action_delta['l1_mean']):.6f} "
                f"delta_l2_norm={float(action_delta['l2_norm']):.6f} "
                f"delta_abs_max={float(action_delta['abs_max']):.6f}"
            )
    print(f"received_actions: {action_chunk.shape[0]}")
    print("first_raw_action:", action_chunk[0].tolist())

    if args.send_mode == "first":
        action_send_dt_ms, cmd_vel = handle_action(action_chunk[0, :3], args, metadata)
        print(f"cmd_send_time_ms: {action_send_dt_ms:.2f}")
        cmd_similarity_metrics = maybe_print_cmd_similarity(cmd_vel, args)
        if args.execution_mode == "task_execution":
            print(f"holding_cmd_vel_sec: {args.execution_horizon_sec:.3f}")
            time.sleep(max(0.0, args.execution_horizon_sec))
            if args.execution_settle_sec > 0.0:
                print(f"settling_after_stop_sec: {args.execution_settle_sec:.3f}")
                time.sleep(args.execution_settle_sec)
        cycle_total_dt_ms = (time.perf_counter() - cycle_t0) * 1000.0
        print(f"cycle_total_time_ms: {cycle_total_dt_ms:.2f}")
        first_raw_action = action_chunk[0].tolist()
        append_cycle_metrics_csv(
            args,
            {
                "wall_time_epoch_s": time.time(),
                "cycle_index": getattr(args, "_current_cycle_index", 1),
                "observation_source": args.observation_source
                or ("snapshot_server" if args.snapshot_server_url else "legacy_robot_observer" if (args.legacy_robot_observer or args.robot_observer) else "json" if args.observation_json else "dummy"),
                "task": args.task,
                "image_height": metadata["image_shape"][0] if metadata else observation["observation.images.front"].shape[0],
                "image_width": metadata["image_shape"][1] if metadata else observation["observation.images.front"].shape[1],
                "state_dim": metadata["state_dim"] if metadata else int(observation["observation.state"].shape[0]),
                "front_obstacle_distance_m": None if metadata is None else metadata.get("front_obstacle_distance_m"),
                "observation_load_time_ms": observation_dt_ms,
                "server_roundtrip_time_ms": response["roundtrip_time_ms"],
                "server_inference_time_ms": response["inference_time_ms"],
                "transport_and_rpc_overhead_ms": transport_overhead_ms,
                "cmd_send_time_ms": action_send_dt_ms,
                "cycle_total_time_ms": cycle_total_dt_ms,
                "execution_mode": args.execution_mode,
                "execution_horizon_sec": args.execution_horizon_sec,
                "execution_settle_sec": args.execution_settle_sec,
                "received_actions": int(action_chunk.shape[0]),
                "first_raw_action_linear_x": first_raw_action[0],
                "first_raw_action_linear_y": first_raw_action[1],
                "first_raw_action_angular_z": first_raw_action[2],
                "final_cmd_linear_x": cmd_vel["linear.x"],
                "final_cmd_linear_y": cmd_vel["linear.y"],
                "final_cmd_angular_z": cmd_vel["angular.z"],
                "image_similarity_l1": None if image_similarity_metrics is None else image_similarity_metrics["image_similarity_l1"],
                "image_similarity_corr": None if image_similarity_metrics is None else image_similarity_metrics["image_similarity_corr"],
                "image_mean_abs_diff": None if image_similarity_metrics is None else image_similarity_metrics["image_mean_abs_diff"],
                "cmd_similarity_cosine": None if cmd_similarity_metrics is None else cmd_similarity_metrics["cmd_similarity_cosine"],
                "cmd_similarity_relative": None if cmd_similarity_metrics is None else cmd_similarity_metrics["cmd_similarity_relative"],
                "cmd_delta_l2": None if cmd_similarity_metrics is None else cmd_similarity_metrics["cmd_delta_l2"],
            },
        )
        return

    period = 1.0 / args.control_rate_hz
    print(f"streaming_chunk_at_hz: {args.control_rate_hz:.2f}")
    last_cmd_vel: dict[str, float] | None = None
    last_cmd_send_dt_ms: float | None = None
    last_cmd_similarity_metrics: dict[str, float] | None = None
    for action in action_chunk:
        t0 = time.perf_counter()
        action_send_dt_ms, cmd_vel = handle_action(action[:3], args, metadata)
        print(f"cmd_send_time_ms: {action_send_dt_ms:.2f}")
        last_cmd_similarity_metrics = maybe_print_cmd_similarity(cmd_vel, args)
        last_cmd_vel = cmd_vel
        last_cmd_send_dt_ms = action_send_dt_ms
        time.sleep(max(0.0, period - (time.perf_counter() - t0)))
    cycle_total_dt_ms = (time.perf_counter() - cycle_t0) * 1000.0
    print(f"cycle_total_time_ms: {cycle_total_dt_ms:.2f}")
    append_cycle_metrics_csv(
        args,
        {
            "wall_time_epoch_s": time.time(),
            "cycle_index": getattr(args, "_current_cycle_index", 1),
            "observation_source": args.observation_source
            or ("snapshot_server" if args.snapshot_server_url else "legacy_robot_observer" if (args.legacy_robot_observer or args.robot_observer) else "json" if args.observation_json else "dummy"),
            "task": args.task,
            "image_height": metadata["image_shape"][0] if metadata else observation["observation.images.front"].shape[0],
            "image_width": metadata["image_shape"][1] if metadata else observation["observation.images.front"].shape[1],
            "state_dim": metadata["state_dim"] if metadata else int(observation["observation.state"].shape[0]),
            "front_obstacle_distance_m": None if metadata is None else metadata.get("front_obstacle_distance_m"),
            "observation_load_time_ms": observation_dt_ms,
            "server_roundtrip_time_ms": response["roundtrip_time_ms"],
            "server_inference_time_ms": response["inference_time_ms"],
            "transport_and_rpc_overhead_ms": transport_overhead_ms,
            "cmd_send_time_ms": last_cmd_send_dt_ms,
            "cycle_total_time_ms": cycle_total_dt_ms,
            "execution_mode": args.execution_mode,
            "execution_horizon_sec": args.execution_horizon_sec,
            "execution_settle_sec": args.execution_settle_sec,
            "received_actions": int(action_chunk.shape[0]),
            "first_raw_action_linear_x": float(action_chunk[0, 0]),
            "first_raw_action_linear_y": float(action_chunk[0, 1]),
            "first_raw_action_angular_z": float(action_chunk[0, 2]),
            "final_cmd_linear_x": None if last_cmd_vel is None else last_cmd_vel["linear.x"],
            "final_cmd_linear_y": None if last_cmd_vel is None else last_cmd_vel["linear.y"],
            "final_cmd_angular_z": None if last_cmd_vel is None else last_cmd_vel["angular.z"],
            "image_similarity_l1": None if image_similarity_metrics is None else image_similarity_metrics["image_similarity_l1"],
            "image_similarity_corr": None if image_similarity_metrics is None else image_similarity_metrics["image_similarity_corr"],
            "image_mean_abs_diff": None if image_similarity_metrics is None else image_similarity_metrics["image_mean_abs_diff"],
            "cmd_similarity_cosine": None if last_cmd_similarity_metrics is None else last_cmd_similarity_metrics["cmd_similarity_cosine"],
            "cmd_similarity_relative": None if last_cmd_similarity_metrics is None else last_cmd_similarity_metrics["cmd_similarity_relative"],
            "cmd_delta_l2": None if last_cmd_similarity_metrics is None else last_cmd_similarity_metrics["cmd_delta_l2"],
        },
    )


def main() -> None:
    args = parse_args()

    if args.safety_slow_distance_m < args.safety_stop_distance_m:
        raise ValueError("--safety-slow-distance-m must be >= --safety-stop-distance-m")
    if args.execution_horizon_sec < 0.0:
        raise ValueError("--execution-horizon-sec must be >= 0")
    if args.execution_settle_sec < 0.0:
        raise ValueError("--execution-settle-sec must be >= 0")

    if args.continuous and args.send_mode == "chunk" and args.actions_per_chunk > 1:
        print(
            "warning: continuous+chunk mode replans only once per returned chunk. "
            "For tighter obstacle avoidance, prefer --send-mode first."
        )
    if args.execution_mode == "task_execution" and args.send_mode != "first":
        print(
            "warning: task_execution mode is designed for --send-mode first. "
            "Chunk streaming still replans only after the whole chunk finishes."
        )

    if not args.continuous:
        args._current_cycle_index = 1
        execute_cycle(args)
        return

    cycle_index = 0
    while args.max_cycles <= 0 or cycle_index < args.max_cycles:
        cycle_index += 1
        args._current_cycle_index = cycle_index
        print(f"continuous_cycle: {cycle_index}")
        execute_cycle(args)


if __name__ == "__main__":
    main()

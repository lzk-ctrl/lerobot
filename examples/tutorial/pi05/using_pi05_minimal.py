#!/usr/bin/env python

"""Minimal PI0.5 inference example with user-defined I/O.

This example is intentionally decoupled from ROS2 and the LeRobot Robot API.
You provide:
  - one RGB image as a NumPy array shaped [H, W, 3], dtype=uint8
  - one state vector as a NumPy array shaped [state_dim], dtype=float32

The script returns one action vector and shows how to interpret the first three
dimensions as a planar cmd_vel command.

Typical workflow for a GO2-like simulator:
  1. Replace `get_observation()` with your own camera/state reader.
  2. Replace `send_action()` with your own `/cmd_vel` publisher or bridge call.
  3. Run the script once to verify that PI0.5 can ingest your observation and
     produce an action tensor end to end.
"""

from __future__ import annotations

import argparse
import copy
import json
import socket
import time
from pathlib import Path

import numpy as np
import torch

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors
from lerobot.policies.utils import prepare_observation_for_inference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal PI0.5 inference example")
    parser.add_argument("--model-id", default="lerobot/pi05_base", help="HF model id or local path")
    parser.add_argument(
        "--device",
        default="auto",
        help="Target inference device, e.g. auto, cpu, mps, cuda, cuda:1",
    )
    parser.add_argument("--task", default="walk forward")
    parser.add_argument("--state-dim", type=int, default=22)
    parser.add_argument("--action-dim", type=int, default=3)
    parser.add_argument("--image-height", type=int, default=480)
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument(
        "--tokenizer-path",
        default=None,
        help="Optional local tokenizer directory. Defaults to <model-id>/tokenizer when present.",
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


def build_config(args: argparse.Namespace, device: torch.device) -> PI05Config:
    dtype = "bfloat16" if device.type == "cuda" else "float32"
    return PI05Config(
        device=str(device),
        dtype=dtype,
        num_inference_steps=args.num_inference_steps,
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


def get_observation(args: argparse.Namespace) -> dict[str, np.ndarray]:
    """Replace this with your own simulator or ROS2 observation reader.

    Contract:
      - image: uint8, shape [H, W, 3]
      - state: float32, shape [state_dim]
      - state values should ideally already be scaled near [-1, 1]

    For a first smoke test we feed zeros, which is enough to verify the full
    PI0.5 inference pipeline.
    """

    image = np.zeros((args.image_height, args.image_width, 3), dtype=np.uint8)
    state = np.zeros((args.state_dim,), dtype=np.float32)

    return {
        "observation.images.front": image,
        "observation.state": state,
    }


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))


def build_cmd_vel(action: np.ndarray, args: argparse.Namespace) -> dict[str, float]:
    return {
        "linear.x": clamp(float(action[0]), -args.max_linear_x, args.max_linear_x),
        "linear.y": clamp(float(action[1]), -args.max_linear_y, args.max_linear_y),
        "angular.z": clamp(float(action[2]), -args.max_angular_z, args.max_angular_z),
    }


def send_udp_cmd_vel(cmd_vel: dict[str, float], args: argparse.Namespace) -> None:
    payload = {
        "source": "pi05_minimal",
        "timestamp": time.time(),
        **cmd_vel,
    }
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.sendto(json.dumps(payload).encode("utf-8"), (args.udp_host, args.udp_port))
    finally:
        sock.close()


def send_action(action: np.ndarray, args: argparse.Namespace) -> None:
    """Replace this with your own output path.

    For a GO2 + CHAMP setup, a common mapping is:
      action[0] -> cmd_vel.linear.x
      action[1] -> cmd_vel.linear.y
      action[2] -> cmd_vel.angular.z
    """

    cmd_vel = build_cmd_vel(action, args)
    if args.cmd_vel_output in {"print", "print+udp"}:
        print("cmd_vel:", cmd_vel)
    if args.cmd_vel_output in {"udp", "print+udp"}:
        send_udp_cmd_vel(cmd_vel, args)
        print(f"sent_udp_cmd_vel: {args.udp_host}:{args.udp_port}")


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


def main() -> None:
    args = parse_args()
    runtime_device = select_device(args.device)
    config = build_config(args, runtime_device)
    load_config = copy.deepcopy(config)
    load_config.device = "cpu"
    tokenizer_path = resolve_tokenizer_path(args.model_id, args.tokenizer_path)

    print(f"Loading policy on cpu from {args.model_id}")
    policy = PI05Policy.from_pretrained(args.model_id, config=load_config)
    if str(runtime_device) != "cpu":
        print(f"Moving policy to {runtime_device}")
        policy.model.to(runtime_device)
    policy.config.device = str(runtime_device)
    print(f"Using tokenizer from {tokenizer_path}")
    preprocess, postprocess = make_pi05_pre_post_processors(
        policy.config,
        tokenizer_name_or_path=tokenizer_path,
    )

    raw_observation = get_observation(args)
    obs = prepare_observation_for_inference(raw_observation, runtime_device, task=args.task)

    synchronize_if_needed(runtime_device)
    t0 = time.perf_counter()
    processed_obs = preprocess(obs)
    action = policy.select_action(processed_obs)
    action = postprocess(action)
    synchronize_if_needed(runtime_device)
    dt_ms = (time.perf_counter() - t0) * 1000.0

    action_np = action.squeeze(0).detach().cpu().numpy()
    print(f"inference_time_ms: {dt_ms:.2f}")
    print("raw_action:", action_np.tolist())

    if len(action_np) >= 3:
        send_action(action_np[:3], args)


if __name__ == "__main__":
    main()

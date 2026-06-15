#!/usr/bin/env python

from __future__ import annotations

import base64
import copy
import html
import json
import math
import pickle
import socket
import struct
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
from lerobot.utils.nvtx import nvtx_range


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


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))


def build_cmd_vel(
    action: np.ndarray,
    max_linear_x: float,
    max_linear_y: float,
    max_angular_z: float,
) -> dict[str, float]:
    return {
        "linear.x": clamp(float(action[0]), -max_linear_x, max_linear_x),
        "linear.y": clamp(float(action[1]), -max_linear_y, max_linear_y),
        "angular.z": clamp(float(action[2]), -max_angular_z, max_angular_z),
    }


def send_udp_cmd_vel(
    cmd_vel: dict[str, float],
    udp_host: str,
    udp_port: int,
    source: str = "pi05_client",
) -> None:
    payload = {
        "source": source,
        "timestamp": time.time(),
        **cmd_vel,
    }
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.sendto(json.dumps(payload).encode("utf-8"), (udp_host, udp_port))
    finally:
        sock.close()


def make_dummy_observation(image_height: int, image_width: int, state_dim: int) -> dict[str, np.ndarray]:
    return {
        "observation.images.front": np.zeros((image_height, image_width, 3), dtype=np.uint8),
        "observation.state": np.zeros((state_dim,), dtype=np.float32),
    }


def _normalize_signed(value: float, scale: float, offset: float = 0.0) -> float:
    if scale <= 0:
        raise ValueError(f"Normalization scale must be positive, received {scale}")
    return float(np.clip((float(value) - offset) / scale, -1.0, 1.0))


def _normalize_positive_range(value: float, lower: float, upper: float) -> float:
    if upper <= lower:
        raise ValueError(f"Invalid range [{lower}, {upper}] for positive-range normalization")
    clipped = float(np.clip(float(value), lower, upper))
    ratio = (clipped - lower) / (upper - lower)
    return ratio * 2.0 - 1.0


def _quaternion_to_euler_xyz(x: float, y: float, z: float, w: float) -> tuple[float, float, float]:
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def decode_robot_observer_image(snapshot: dict[str, Any]) -> np.ndarray:
    image_msg = snapshot.get("messages", {}).get("camera_image")
    if image_msg is None:
        raise ValueError("Snapshot is missing messages.camera_image")

    image_base64 = image_msg.get("data_base64")
    if not image_base64:
        raise ValueError(
            "Snapshot does not include camera image bytes. "
            "Generate it with robot_observer.py --include-image-data."
        )

    height = int(image_msg["height"])
    width = int(image_msg["width"])
    encoding = str(image_msg["encoding"]).lower()

    if encoding not in {"rgb8", "bgr8"}:
        raise ValueError(f"Unsupported camera encoding: {encoding}. Expected rgb8 or bgr8.")

    image_bytes = base64.b64decode(image_base64)
    image = np.frombuffer(image_bytes, dtype=np.uint8)
    expected_size = height * width * 3
    if image.size != expected_size:
        raise ValueError(
            f"Decoded image size mismatch. Expected {expected_size} bytes, received {image.size}."
        )

    image = image.reshape(height, width, 3).copy()
    if encoding == "bgr8":
        image = image[:, :, ::-1].copy()

    return image


def extract_front_obstacle_distance(snapshot: dict[str, Any]) -> float | None:
    messages = snapshot.get("messages", {})
    for key in ["front_obstacle_distance", "front_obstacle_distance_raw"]:
        msg = messages.get(key)
        if msg is None:
            continue
        value = msg.get("data")
        if value is None:
            continue
        value = float(value)
        if math.isfinite(value):
            return value
    return None


def build_champ_state_vector(snapshot: dict[str, Any]) -> tuple[np.ndarray, list[str]]:
    messages = snapshot.get("messages", {})

    joint_states = messages.get("joint_states")
    foot_contacts = messages.get("foot_contacts")
    imu = messages.get("imu")
    base_pose = messages.get("base_to_footprint_pose")

    if joint_states is None:
        raise ValueError("Snapshot is missing messages.joint_states")
    if foot_contacts is None:
        raise ValueError("Snapshot is missing messages.foot_contacts")
    if imu is None:
        raise ValueError("Snapshot is missing messages.imu")
    if base_pose is None:
        raise ValueError("Snapshot is missing messages.base_to_footprint_pose")

    joint_names = list(joint_states["name"])
    joint_positions = list(joint_states["position"])
    joint_velocities = list(joint_states["velocity"])
    contacts = list(foot_contacts["contacts"])

    if len(joint_names) != len(joint_positions) or len(joint_names) != len(joint_velocities):
        raise ValueError(
            "joint_states name/position/velocity lengths do not match: "
            f"{len(joint_names)=}, {len(joint_positions)=}, {len(joint_velocities)=}"
        )

    if len(contacts) != 4:
        raise ValueError(f"Expected 4 foot contact values, received {len(contacts)}")

    orientation = imu["orientation"]
    angular_velocity = imu["angular_velocity"]
    front_obstacle_distance = extract_front_obstacle_distance(snapshot)
    roll, pitch, _yaw = _quaternion_to_euler_xyz(
        float(orientation["x"]),
        float(orientation["y"]),
        float(orientation["z"]),
        float(orientation["w"]),
    )
    yaw_rate = float(angular_velocity["z"])

    if front_obstacle_distance is None:
        # Treat missing range as "no close obstacle detected" while keeping the vector length fixed.
        front_obstacle_distance = 2.0

    state_values: list[float] = []
    state_names: list[str] = []

    for joint_name, joint_position in zip(joint_names, joint_positions, strict=True):
        state_names.append(f"{joint_name}.pos")
        state_values.append(_normalize_signed(joint_position, scale=np.pi))

    for joint_name, joint_velocity in zip(joint_names, joint_velocities, strict=True):
        state_names.append(f"{joint_name}.vel")
        state_values.append(_normalize_signed(joint_velocity, scale=10.0))

    for idx, is_in_contact in enumerate(contacts):
        state_names.append(f"foot_contact_{idx}")
        state_values.append(1.0 if bool(is_in_contact) else -1.0)

    state_names.extend(["imu.roll", "imu.pitch", "imu.yaw_rate", "front_obstacle_distance"])
    state_values.extend(
        [
            _normalize_signed(roll, scale=0.5),
            _normalize_signed(pitch, scale=0.5),
            _normalize_signed(yaw_rate, scale=2.0),
            _normalize_positive_range(front_obstacle_distance, lower=0.0, upper=2.0),
        ]
    )

    state = np.asarray(state_values, dtype=np.float32)
    return state, state_names


def build_pi05_observation_from_robot_observer_snapshot(
    snapshot: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    messages = snapshot.get("messages", {})
    camera_snapshot = messages.get("camera_image")
    image = decode_robot_observer_image(snapshot)
    state, state_names = build_champ_state_vector(snapshot)
    observation = {
        "observation.images.front": image,
        "observation.state": state,
    }
    metadata = {
        "image_shape": tuple(int(dim) for dim in image.shape),
        "state_dim": int(state.shape[0]),
        "state_names": state_names,
        "state_values": state.tolist(),
        "front_obstacle_distance_m": extract_front_obstacle_distance(snapshot),
        "camera_snapshot": camera_snapshot,
        "raw_messages": {
            "joint_states": messages.get("joint_states"),
            "foot_contacts": messages.get("foot_contacts"),
            "imu": messages.get("imu"),
            "base_to_footprint_pose": messages.get("base_to_footprint_pose"),
            "odom": messages.get("odom"),
            "odom_local": messages.get("odom_local"),
            "front_obstacle_distance": messages.get("front_obstacle_distance"),
            "front_obstacle_distance_raw": messages.get("front_obstacle_distance_raw"),
        },
    }
    return observation, metadata


def build_pi05_observation_from_robot_observer_json(
    snapshot_path: str | Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    snapshot_path = Path(snapshot_path).expanduser()
    with snapshot_path.open() as f:
        snapshot = json.load(f)
    return build_pi05_observation_from_robot_observer_snapshot(snapshot)


def save_observation_debug_html(
    output_path: str | Path,
    metadata: dict[str, Any],
    task: str | None = None,
) -> Path:
    output_path = Path(output_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    camera_snapshot = metadata.get("camera_snapshot") or {}
    state_names = list(metadata.get("state_names", []))
    state_values = list(metadata.get("state_values", []))
    raw_messages = metadata.get("raw_messages", {})
    front_obstacle_distance = metadata.get("front_obstacle_distance_m")

    state_rows = "\n".join(
        f"<tr><td>{idx}</td><td>{html.escape(str(name))}</td><td>{float(value):.6f}</td></tr>"
        for idx, (name, value) in enumerate(zip(state_names, state_values, strict=False))
    )
    raw_messages_html = html.escape(json.dumps(raw_messages, ensure_ascii=False, indent=2))
    camera_payload_json = json.dumps(camera_snapshot, ensure_ascii=False)
    front_obstacle_label = (
        "unavailable" if front_obstacle_distance is None else f"{float(front_obstacle_distance):.6f} m"
    )
    task_label = html.escape(task or "")

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>First Observation</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      margin: 24px;
      background: #f6f7fb;
      color: #1f2937;
    }}
    h1, h2 {{
      margin: 0 0 12px 0;
    }}
    .grid {{
      display: grid;
      grid-template-columns: minmax(320px, 720px) minmax(320px, 1fr);
      gap: 20px;
      align-items: start;
    }}
    .card {{
      background: white;
      border: 1px solid #dbe2ea;
      border-radius: 12px;
      padding: 16px;
      box-shadow: 0 4px 18px rgba(15, 23, 42, 0.06);
    }}
    canvas {{
      width: 100%;
      height: auto;
      border-radius: 8px;
      border: 1px solid #dbe2ea;
      background: #111827;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    th, td {{
      text-align: left;
      padding: 6px 8px;
      border-bottom: 1px solid #edf2f7;
      vertical-align: top;
    }}
    pre {{
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      font-size: 13px;
      line-height: 1.45;
    }}
    .meta {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px 16px;
      margin-bottom: 16px;
      font-size: 14px;
    }}
    .meta strong {{
      display: inline-block;
      min-width: 180px;
    }}
  </style>
</head>
<body>
  <h1>First Observation Snapshot</h1>
  <div class="meta">
    <div><strong>Task</strong>{task_label}</div>
    <div><strong>Image Shape</strong>{html.escape(str(metadata.get("image_shape")))}</div>
    <div><strong>State Dim</strong>{int(metadata.get("state_dim", 0))}</div>
    <div><strong>Front Obstacle Distance</strong>{html.escape(front_obstacle_label)}</div>
  </div>

  <div class="grid">
    <section class="card">
      <h2>Camera</h2>
      <canvas id="camera-canvas"></canvas>
    </section>

    <section class="card">
      <h2>Normalized State</h2>
      <table>
        <thead>
          <tr><th>#</th><th>Name</th><th>Value</th></tr>
        </thead>
        <tbody>
          {state_rows}
        </tbody>
      </table>
    </section>
  </div>

  <section class="card" style="margin-top:20px;">
    <h2>Raw Messages</h2>
    <pre>{raw_messages_html}</pre>
  </section>

  <script>
    const cameraPayload = {camera_payload_json};
    const canvas = document.getElementById("camera-canvas");
    const ctx = canvas.getContext("2d");
    if (cameraPayload && cameraPayload.data_base64) {{
      const width = cameraPayload.width;
      const height = cameraPayload.height;
      const encoding = (cameraPayload.encoding || "rgb8").toLowerCase();
      const raw = Uint8Array.from(atob(cameraPayload.data_base64), c => c.charCodeAt(0));
      const rgba = new Uint8ClampedArray(width * height * 4);
      for (let src = 0, dst = 0; src + 2 < raw.length; src += 3, dst += 4) {{
        if (encoding === "bgr8") {{
          rgba[dst] = raw[src + 2];
          rgba[dst + 1] = raw[src + 1];
          rgba[dst + 2] = raw[src];
        }} else {{
          rgba[dst] = raw[src];
          rgba[dst + 1] = raw[src + 1];
          rgba[dst + 2] = raw[src + 2];
        }}
        rgba[dst + 3] = 255;
      }}
      canvas.width = width;
      canvas.height = height;
      ctx.putImageData(new ImageData(rgba, width, height), 0, 0);
    }} else {{
      canvas.width = 640;
      canvas.height = 480;
      ctx.fillStyle = "#111827";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = "#f9fafb";
      ctx.font = "20px sans-serif";
      ctx.fillText("No camera image available", 20, 40);
    }}
  </script>
</body>
</html>
"""
    output_path.write_text(html_doc, encoding="utf-8")
    return output_path


def validate_observation(
    observation: dict[str, Any],
    image_height: int,
    image_width: int,
    max_state_dim: int,
) -> dict[str, np.ndarray]:
    required_keys = {"observation.images.front", "observation.state"}
    missing_keys = required_keys - set(observation)
    if missing_keys:
        raise ValueError(f"Observation is missing required keys: {sorted(missing_keys)}")

    image = np.asarray(observation["observation.images.front"], dtype=np.uint8)
    state = np.asarray(observation["observation.state"], dtype=np.float32)

    expected_image_shape = (image_height, image_width, 3)
    if image.shape != expected_image_shape:
        raise ValueError(f"Image shape must be {expected_image_shape}, received {image.shape}")

    if state.ndim != 1:
        raise ValueError(f"State must be a 1D vector, received shape {state.shape}")

    if state.shape[0] > max_state_dim:
        raise ValueError(
            f"State length must be <= {max_state_dim}, received {state.shape[0]}. "
            "Trim or redesign the state vector before inference."
        )

    return {
        "observation.images.front": image,
        "observation.state": state,
    }


def send_message(sock: socket.socket, payload: Any) -> None:
    data = pickle.dumps(payload)  # nosec B301: local trusted example usage
    header = struct.pack("!Q", len(data))
    sock.sendall(header + data)


def recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = sock.recv(size - len(chunks))
        if not chunk:
            raise EOFError("Connection closed while receiving data")
        chunks.extend(chunk)
    return bytes(chunks)


def recv_message(sock: socket.socket) -> Any:
    header = recv_exact(sock, 8)
    size = struct.unpack("!Q", header)[0]
    data = recv_exact(sock, size)
    return pickle.loads(data)  # nosec B301: local trusted example usage


def _compute_tensor_delta_metrics(tensor: torch.Tensor) -> dict[str, float]:
    tensor = tensor.to(dtype=torch.float32)
    return {
        "l1_mean": float(tensor.abs().mean().item()),
        "l2_norm": float(torch.linalg.vector_norm(tensor).item()),
        "abs_max": float(tensor.abs().max().item()),
    }


def _build_denoising_debug_artifact(
    *,
    noise: torch.Tensor,
    tracked_steps: list,
    num_inference_steps: int,
    action_dim: int,
) -> dict[str, Any]:
    if not tracked_steps:
        raise RuntimeError("Denoising debug is enabled but no denoising steps were captured")

    ordered_steps = sorted(tracked_steps, key=lambda step: step.step_idx)
    if len(ordered_steps) != num_inference_steps:
        raise RuntimeError(
            "Denoising debug step count mismatch: "
            f"expected {num_inference_steps}, captured {len(ordered_steps)}"
        )

    noise_cpu = noise.detach().cpu().to(dtype=torch.float32)
    if noise_cpu.ndim == 4:
        noise_for_steps = noise_cpu.reshape(
            noise_cpu.shape[0] * noise_cpu.shape[1],
            noise_cpu.shape[2],
            noise_cpu.shape[3],
        )
    else:
        noise_for_steps = noise_cpu
    previous_x_after = noise_for_steps
    step_payloads: list[dict[str, Any]] = []
    step_summaries: list[dict[str, Any]] = []

    for step in ordered_steps:
        if step.x_t is None or step.v_t is None:
            raise RuntimeError(f"Tracker step {step.step_idx} is missing x_t or v_t")

        x_before = previous_x_after
        x_after = step.x_t.detach().cpu().to(dtype=torch.float32)
        v_t = step.v_t.detach().cpu().to(dtype=torch.float32)
        delta_t = x_after - x_before
        expected_delta_t = (-1.0 / num_inference_steps) * v_t

        x_before_action = x_before[..., :action_dim]
        x_after_action = x_after[..., :action_dim]
        v_t_action = v_t[..., :action_dim]
        delta_t_action = delta_t[..., :action_dim]
        expected_delta_action = expected_delta_t[..., :action_dim]

        step_payloads.append(
            {
                "step_idx": int(step.step_idx),
                "time": float(step.time),
                "x_t_before": x_before.clone(),
                "x_t_after": x_after.clone(),
                "v_t": v_t.clone(),
                "delta_t": delta_t.clone(),
                "expected_delta_t": expected_delta_t.clone(),
                "x_t_before_action": x_before_action.clone(),
                "x_t_after_action": x_after_action.clone(),
                "v_t_action": v_t_action.clone(),
                "delta_t_action": delta_t_action.clone(),
            }
        )
        step_summaries.append(
            {
                "step_idx": int(step.step_idx),
                "time": float(step.time),
                "full_delta": _compute_tensor_delta_metrics(delta_t),
                "action_delta": _compute_tensor_delta_metrics(delta_t_action),
                "action_velocity": _compute_tensor_delta_metrics(v_t_action),
                "action_state_after": _compute_tensor_delta_metrics(x_after_action),
                "delta_vs_expected_abs_max": float(
                    (delta_t_action - expected_delta_action).abs().max().item()
                ),
            }
        )
        previous_x_after = x_after

    return {
        "num_inference_steps": int(num_inference_steps),
        "action_dim": int(action_dim),
        "internal_action_dim": int(noise_cpu.shape[-1]),
        "chunk_size": int(noise_cpu.shape[-2]),
        "noise": noise_cpu,
        "noise_action": noise_cpu[..., :action_dim].clone(),
        "steps": step_payloads,
        "step_summaries": step_summaries,
    }


def _load_noise_tensor(
    noise_path: Path,
    expected_shape: tuple[int, ...],
    device: torch.device,
) -> torch.Tensor:
    payload = torch.load(noise_path, map_location="cpu")
    if isinstance(payload, dict):
        if "noise" not in payload:
            raise ValueError(f"Noise file {noise_path} is missing required 'noise' tensor")
        tensor = payload["noise"]
    else:
        tensor = payload

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Noise file {noise_path} must contain a torch.Tensor, got {type(tensor)!r}")

    tensor = tensor.detach().clone().to(dtype=torch.float32)
    if tensor.ndim == len(expected_shape) - 1 and expected_shape[0] == 1:
        tensor = tensor.unsqueeze(0)

    if tuple(tensor.shape) != expected_shape:
        raise ValueError(
            f"Noise tensor shape mismatch for {noise_path}: expected {expected_shape}, got {tuple(tensor.shape)}"
        )

    return tensor.to(device=device)


def _save_noise_tensor(
    noise_path: Path,
    noise: torch.Tensor,
    *,
    model_id: str,
    num_inference_steps: int,
    source: str,
) -> None:
    noise_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "noise": noise.detach().cpu().to(dtype=torch.float32),
            "model_id": model_id,
            "num_inference_steps": int(num_inference_steps),
            "source": source,
            "saved_at_epoch_s": float(time.time()),
        },
        noise_path,
    )


def _compute_warm_start_cmd_metrics(
    *,
    initial_noise: torch.Tensor,
    action_dim: int,
    postprocess,
    final_action_tensor: torch.Tensor,
) -> dict[str, Any]:
    if initial_noise.ndim == 4:
        initial_action = initial_noise[:, 0, 0, :action_dim]
    else:
        initial_action = initial_noise[:, 0, :action_dim]
    initial_cmd_tensor = postprocess(initial_action).squeeze(0).detach().cpu().to(dtype=torch.float32)
    if final_action_tensor.ndim == 3:
        final_cmd_tensor = final_action_tensor[0, 0].detach().cpu().to(dtype=torch.float32)
    else:
        final_cmd_tensor = final_action_tensor[0].detach().cpu().to(dtype=torch.float32)
    cmd_delta = final_cmd_tensor - initial_cmd_tensor
    return {
        "initial_cmd": initial_cmd_tensor.tolist(),
        "final_cmd": final_cmd_tensor.tolist(),
        "cmd_delta_xyz": cmd_delta.tolist(),
        "cmd_delta_l2": float(torch.linalg.vector_norm(cmd_delta).item()),
    }


class PI05InferenceEngine:
    def __init__(
        self,
        model_id: str,
        device: str = "auto",
        state_dim: int = 22,
        action_dim: int = 3,
        image_height: int = 480,
        image_width: int = 640,
        chunk_size: int = 50,
        num_inference_steps: int = 10,
        num_action_samples: int = 1,
        compile_model: bool = False,
        tokenizer_path: str | None = None,
        denoising_debug_dir: str | None = None,
        reusable_noise_path: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.runtime_device = select_device(device)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.image_height = image_height
        self.image_width = image_width
        self.tokenizer_path = resolve_tokenizer_path(model_id, tokenizer_path)
        self.lock = threading.Lock()
        self.denoising_debug_dir = Path(denoising_debug_dir).expanduser() if denoising_debug_dir else None
        self.reusable_noise_path = Path(reusable_noise_path).expanduser() if reusable_noise_path else None
        self._denoising_debug_request_idx = 0

        rtc_config = None
        if self.denoising_debug_dir is not None:
            self.denoising_debug_dir.mkdir(parents=True, exist_ok=True)
            rtc_config = RTCConfig(
                enabled=False,
                debug=True,
                debug_maxlen=max(100, num_inference_steps + 1),
            )
        if self.reusable_noise_path is not None:
            self.reusable_noise_path.parent.mkdir(parents=True, exist_ok=True)

        dtype = "bfloat16" if self.runtime_device.type == "cuda" else "float32"
        self.config = PI05Config(
            device=str(self.runtime_device),
            dtype=dtype,
            chunk_size=chunk_size,
            n_action_steps=chunk_size,
            num_inference_steps=num_inference_steps,
            num_action_samples=num_action_samples,
            compile_model=compile_model,
            rtc_config=rtc_config,
            input_features={
                "observation.images.front": PolicyFeature(
                    type=FeatureType.VISUAL,
                    shape=(3, image_height, image_width),
                ),
                "observation.state": PolicyFeature(
                    type=FeatureType.STATE,
                    shape=(state_dim,),
                ),
            },
            output_features={
                "action": PolicyFeature(
                    type=FeatureType.ACTION,
                    shape=(action_dim,),
                )
            },
            normalization_mapping={
                "VISUAL": NormalizationMode.IDENTITY,
                "STATE": NormalizationMode.IDENTITY,
                "ACTION": NormalizationMode.IDENTITY,
            },
        )

        load_config = copy.deepcopy(self.config)
        load_config.device = "cpu"

        print(f"Loading policy on cpu from {self.model_id}")
        self.policy = PI05Policy.from_pretrained(self.model_id, config=load_config)
        if str(self.runtime_device) != "cpu":
            print(f"Moving policy to {self.runtime_device}")
            self.policy.model.to(self.runtime_device)
        self.policy.config.device = str(self.runtime_device)
        print(f"Using tokenizer from {self.tokenizer_path}")
        self.preprocess, self.postprocess = make_pi05_pre_post_processors(
            self.policy.config,
            tokenizer_name_or_path=self.tokenizer_path,
        )

    def _save_denoising_debug_artifact(
        self,
        artifact: dict[str, Any],
        inference_time_ms: float,
    ) -> dict[str, Any]:
        if self.denoising_debug_dir is None:
            raise RuntimeError("Denoising debug directory is not configured")

        self._denoising_debug_request_idx += 1
        request_idx = self._denoising_debug_request_idx
        base_name = f"denoising_debug_{request_idx:06d}"
        artifact_path = self.denoising_debug_dir / f"{base_name}.pt"
        summary_path = self.denoising_debug_dir / f"{base_name}.json"

        torch.save(
            {
                "model_id": self.model_id,
                "device": str(self.runtime_device),
                "inference_time_ms": float(inference_time_ms),
                **artifact,
            },
            artifact_path,
        )

        summary_payload = {
            "request_index": int(request_idx),
            "model_id": self.model_id,
            "device": str(self.runtime_device),
            "inference_time_ms": float(inference_time_ms),
            "num_inference_steps": int(artifact["num_inference_steps"]),
            "action_dim": int(artifact["action_dim"]),
            "internal_action_dim": int(artifact["internal_action_dim"]),
            "chunk_size": int(artifact["chunk_size"]),
            "artifact_path": str(artifact_path),
            "summary_path": str(summary_path),
            "step_summaries": artifact["step_summaries"],
        }
        summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
        return summary_payload

    def infer(
        self,
        observation: dict[str, Any],
        task: str,
        actions_per_chunk: int = 1,
        num_action_samples: int | None = None,
        initial_noise_path: str | None = None,
        save_final_noise_path: str | None = None,
    ) -> dict[str, Any]:
        validated_observation = validate_observation(
            observation,
            image_height=self.image_height,
            image_width=self.image_width,
            max_state_dim=self.policy.config.max_state_dim,
        )

        requested_actions = max(1, min(actions_per_chunk, self.policy.config.chunk_size))
        if num_action_samples is None:
            num_action_samples = self.policy.config.num_action_samples
        num_action_samples = int(num_action_samples)
        if num_action_samples < 1:
            raise ValueError(f"num_action_samples must be >= 1, got {num_action_samples}")
        request_noise_path = Path(initial_noise_path).expanduser() if initial_noise_path else None
        request_save_noise_path = Path(save_final_noise_path).expanduser() if save_final_noise_path else None

        with self.lock:
            with nvtx_range("pi05.engine.infer"):
                with nvtx_range("pi05.engine.prepare_observation_for_inference"):
                    obs = prepare_observation_for_inference(
                        validated_observation,
                        self.runtime_device,
                        task=task,
                    )

                synchronize_if_needed(self.runtime_device)
                t0 = time.perf_counter()

                with nvtx_range("pi05.engine.preprocess"):
                    processed_obs = self.preprocess(obs)

                needs_debug_tracking = self.denoising_debug_dir is not None
                effective_load_noise_path = request_noise_path
                effective_save_noise_path = request_save_noise_path
                if effective_load_noise_path is None:
                    effective_load_noise_path = self.reusable_noise_path
                if effective_save_noise_path is None:
                    effective_save_noise_path = self.reusable_noise_path

                needs_full_internal_action = (
                    needs_debug_tracking
                    or effective_load_noise_path is not None
                    or effective_save_noise_path is not None
                )
                initial_noise = None
                noise_source = "model_random"

                if needs_debug_tracking:
                    if self.policy.rtc_processor is None or not self.policy.rtc_processor.is_debug_enabled():
                        raise RuntimeError("Denoising debug was requested but RTC tracker is not enabled")

                    self.policy.rtc_processor.reset_tracker()

                full_action_chunk = None
                if needs_full_internal_action:
                    batch_size = int(processed_obs["observation.state"].shape[0])
                    if num_action_samples == 1:
                        noise_shape = (
                            batch_size,
                            self.policy.config.chunk_size,
                            self.policy.config.max_action_dim,
                        )
                    else:
                        noise_shape = (
                            batch_size,
                            num_action_samples,
                            self.policy.config.chunk_size,
                            self.policy.config.max_action_dim,
                        )
                    if effective_load_noise_path is not None:
                        if not effective_load_noise_path.is_file():
                            raise FileNotFoundError(
                                f"Initial noise file not found: {effective_load_noise_path}"
                            )
                        initial_noise = _load_noise_tensor(
                            effective_load_noise_path,
                            expected_shape=noise_shape,
                            device=self.runtime_device,
                        )
                        noise_source = "provided_file"
                    else:
                        initial_noise = self.policy.model.sample_noise(noise_shape, self.runtime_device)
                        noise_source = "random_sampled"

                with nvtx_range("pi05.engine.predict_action_chunk"):
                    predict_kwargs: dict[str, Any] = {}
                    if initial_noise is not None:
                        predict_kwargs["noise"] = initial_noise
                    predict_kwargs["num_action_samples"] = num_action_samples

                    if needs_full_internal_action:
                        images, img_masks = self.policy._preprocess_images(processed_obs)
                        tokens = processed_obs[OBS_LANGUAGE_TOKENS]
                        masks = processed_obs[OBS_LANGUAGE_ATTENTION_MASK]
                        full_action_chunk = self.policy.model.sample_actions(
                            images,
                            img_masks,
                            tokens,
                            masks,
                            **predict_kwargs,
                        )
                        original_action_dim = self.policy.config.output_features[ACTION].shape[0]
                        if full_action_chunk.ndim == 4:
                            action_chunk = full_action_chunk[:, :, :requested_actions, :original_action_dim]
                        else:
                            action_chunk = full_action_chunk[:, :requested_actions, :original_action_dim]
                    else:
                        action_chunk = self.policy.predict_action_chunk(processed_obs, **predict_kwargs)
                        if action_chunk.ndim == 4:
                            action_chunk = action_chunk[:, :, :requested_actions, :]
                        else:
                            action_chunk = action_chunk[:, :requested_actions, :]

                with nvtx_range("pi05.engine.postprocess"):
                    if action_chunk.ndim == 4:
                        processed_samples = []
                        for sample_idx in range(action_chunk.shape[1]):
                            processed_actions = []
                            for idx in range(action_chunk.shape[2]):
                                processed_action = self.postprocess(action_chunk[:, sample_idx, idx, :])
                                processed_actions.append(processed_action)
                            processed_samples.append(torch.stack(processed_actions, dim=1))
                        action_tensor = torch.stack(processed_samples, dim=1).squeeze(0)
                    else:
                        processed_actions = []
                        for idx in range(action_chunk.shape[1]):
                            processed_action = self.postprocess(action_chunk[:, idx, :])
                            processed_actions.append(processed_action)

                        action_tensor = torch.stack(processed_actions, dim=1).squeeze(0)

                synchronize_if_needed(self.runtime_device)
                dt_ms = (time.perf_counter() - t0) * 1000.0

        with nvtx_range("pi05.engine.to_numpy"):
            action_np = action_tensor.detach().cpu().numpy()
        result = {
            "action_chunk": action_np,
            "inference_time_ms": dt_ms,
            "actions_per_chunk": int(action_np.shape[-2]),
            "action_dim": int(action_np.shape[-1]) if action_np.ndim >= 2 else 0,
            "num_action_samples": int(num_action_samples),
            "device": str(self.runtime_device),
            "model_id": self.model_id,
        }
        print(
            "inference_time_ms: "
            f"{dt_ms:.2f} "
            f"(device={self.runtime_device}, "
            f"num_inference_steps={self.policy.config.num_inference_steps}, "
            f"num_action_samples={num_action_samples}, "
            f"actions_per_chunk={result['actions_per_chunk']})",
            flush=True,
        )
        if request_noise_path is not None:
            result["noise_input_path"] = str(request_noise_path)

        if initial_noise is not None and noise_source == "provided_file":
            result["warm_start_cmd_metrics"] = _compute_warm_start_cmd_metrics(
                initial_noise=initial_noise,
                action_dim=result["action_dim"],
                postprocess=self.postprocess,
                final_action_tensor=action_tensor,
            )

        if effective_save_noise_path is not None:
            if full_action_chunk is None:
                raise RuntimeError("Noise save path is configured but no full action chunk was produced")
            _save_noise_tensor(
                effective_save_noise_path,
                full_action_chunk,
                model_id=self.model_id,
                num_inference_steps=self.policy.config.num_inference_steps,
                source="final_denoised_result",
            )
            result["noise_source"] = noise_source
            result["saved_noise_path"] = str(effective_save_noise_path)
        elif initial_noise is not None:
            result["noise_source"] = noise_source

        if initial_noise is not None and needs_debug_tracking:
            debug_artifact = _build_denoising_debug_artifact(
                noise=initial_noise,
                tracked_steps=self.policy.rtc_processor.get_all_debug_steps(),
                num_inference_steps=self.policy.config.num_inference_steps,
                action_dim=result["action_dim"],
            )
            result["denoising_debug"] = self._save_denoising_debug_artifact(debug_artifact, dt_ms)
        return result

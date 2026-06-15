#!/usr/bin/env python3

from __future__ import annotations

import argparse
import base64
import json
import threading
import time
from copy import deepcopy
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import rclpy
from champ_msgs.msg import ContactsStamped
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
from rclpy.qos import qos_profile_sensor_data
from rosgraph_msgs.msg import Clock
from rosidl_runtime_py.convert import message_to_ordereddict
from sensor_msgs.msg import Image
from sensor_msgs.msg import Imu
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32


def clock_qos() -> QoSProfile:
    profile = QoSProfile(depth=10)
    profile.reliability = ReliabilityPolicy.BEST_EFFORT
    return profile


class SnapshotObserver(Node):
    TOPICS = {
        "clock": ("/clock", Clock, clock_qos()),
        "joint_states": ("/joint_states", JointState, qos_profile_sensor_data),
        "foot_contacts": ("/foot_contacts", ContactsStamped, 10),
        "imu": ("/imu/data", Imu, qos_profile_sensor_data),
        "odom": ("/odom", Odometry, 10),
        "odom_local": ("/odom/local", Odometry, 10),
        "base_to_footprint_pose": ("/base_to_footprint_pose", PoseWithCovarianceStamped, 10),
        "camera_image": ("/camera/image_raw", Image, qos_profile_sensor_data),
        "front_obstacle_distance": ("/front_obstacle_distance", Float32, 10),
        "front_obstacle_distance_raw": ("/front_obstacle_distance_raw", Float32, 10),
    }
    REQUIRED_KEYS = {
        "joint_states",
        "foot_contacts",
        "imu",
        "base_to_footprint_pose",
        "camera_image",
    }

    def __init__(self) -> None:
        self._owns_rclpy = not rclpy.ok()
        if self._owns_rclpy:
            rclpy.init()

        super().__init__("pi05_observation_snapshot_server")
        self._latest_messages = {name: None for name in self.TOPICS}
        self._latest_wall_time = {name: None for name in self.TOPICS}
        self._subscriptions = []
        self._condition = threading.Condition()

        for name, (topic, msg_type, qos) in self.TOPICS.items():
            self._subscriptions.append(
                self.create_subscription(msg_type, topic, self._make_callback(name), qos)
            )

    def _make_callback(self, name: str):
        def callback(msg) -> None:
            with self._condition:
                self._latest_messages[name] = msg
                self._latest_wall_time[name] = time.time()
                self._condition.notify_all()

        return callback

    def has_required_messages(self) -> bool:
        return all(self._latest_messages[key] is not None for key in self.REQUIRED_KEYS)

    def wait_for_required_messages(self, timeout_sec: float) -> bool:
        deadline = time.monotonic() + max(timeout_sec, 0.0)
        with self._condition:
            while not self.has_required_messages():
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    break
                self._condition.wait(timeout=remaining)
            return self.has_required_messages()

    def _serialize_message(self, msg, include_image_data: bool):
        if msg is None:
            return None

        if isinstance(msg, Image):
            image = {
                "header": message_to_ordereddict(msg.header),
                "height": msg.height,
                "width": msg.width,
                "encoding": msg.encoding,
                "is_bigendian": msg.is_bigendian,
                "step": msg.step,
                "data_length": len(msg.data),
            }
            if include_image_data:
                image["data_base64"] = base64.b64encode(bytes(msg.data)).decode("ascii")
            return image

        return message_to_ordereddict(msg)

    def get_snapshot(self, include_image_data: bool, wait_timeout_sec: float) -> dict:
        self.wait_for_required_messages(wait_timeout_sec)

        with self._condition:
            messages = deepcopy(self._latest_messages)
            updated_at = dict(self._latest_wall_time)

        serialized = {
            name: self._serialize_message(msg, include_image_data)
            for name, msg in messages.items()
        }
        received = {name: msg is not None for name, msg in messages.items()}
        return {
            "received": received,
            "updated_at": updated_at,
            "messages": serialized,
        }

    def close(self) -> None:
        self.destroy_node()
        if self._owns_rclpy and rclpy.ok():
            rclpy.shutdown()


class SnapshotRequestHandler(BaseHTTPRequestHandler):
    observer: SnapshotObserver | None = None
    default_wait_timeout_sec = 0.25

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            self._write_json(
                {
                    "ok": True,
                    "required_ready": self.observer.has_required_messages(),
                }
            )
            return

        if parsed.path != "/latest":
            self.send_error(HTTPStatus.NOT_FOUND, "Unknown endpoint")
            return

        query = parse_qs(parsed.query)
        include_image_data = query.get("include_image_data", ["1"])[0] != "0"
        wait_timeout_sec = float(query.get("timeout", [str(self.default_wait_timeout_sec)])[0])
        snapshot = self.observer.get_snapshot(
            include_image_data=include_image_data,
            wait_timeout_sec=wait_timeout_sec,
        )
        self._write_json(snapshot)

    def log_message(self, format: str, *args) -> None:
        return

    def _write_json(self, payload: dict) -> None:
        data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def spin_observer(observer: SnapshotObserver, shutdown_event: threading.Event) -> None:
    while not shutdown_event.is_set() and rclpy.ok():
        rclpy.spin_once(observer, timeout_sec=0.1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the latest ROS observation snapshot over HTTP.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7001)
    parser.add_argument("--default-wait-timeout-sec", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    observer = SnapshotObserver()
    shutdown_event = threading.Event()
    spin_thread = threading.Thread(target=spin_observer, args=(observer, shutdown_event), daemon=True)
    spin_thread.start()

    SnapshotRequestHandler.observer = observer
    SnapshotRequestHandler.default_wait_timeout_sec = args.default_wait_timeout_sec
    server = ThreadingHTTPServer((args.host, args.port), SnapshotRequestHandler)
    print(f"Observation snapshot server listening on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        shutdown_event.set()
        server.server_close()
        spin_thread.join(timeout=1.0)
        observer.close()


if __name__ == "__main__":
    main()

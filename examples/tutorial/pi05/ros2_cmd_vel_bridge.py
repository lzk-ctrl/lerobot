#!/usr/bin/env python3

"""Receive cmd_vel commands over UDP and publish them to ROS2."""

from __future__ import annotations

import argparse
import json
import socket

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UDP to ROS2 /cmd_vel bridge")
    parser.add_argument("--bind-host", default="127.0.0.1")
    parser.add_argument("--udp-port", type=int, default=5555)
    parser.add_argument("--cmd-vel-topic", default="/cmd_vel")
    parser.add_argument("--poll-rate-hz", type=float, default=50.0)
    parser.add_argument(
        "--command-timeout-sec",
        type=float,
        default=0.35,
        help="Deprecated and ignored. The bridge now forwards commands without injecting timeout zeros.",
    )
    parser.add_argument(
        "--enforce-limits",
        action="store_true",
        help="Clamp received cmd_vel values before publishing. Disabled by default for pure forwarding.",
    )
    parser.add_argument("--max-linear-x", type=float, default=0.3)
    parser.add_argument("--max-linear-y", type=float, default=0.25)
    parser.add_argument("--max-angular-z", type=float, default=0.5)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))


class UdpCmdVelBridge(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("pi05_cmd_vel_bridge")
        self.args = args
        self.publisher = self.create_publisher(Twist, args.cmd_vel_topic, 10)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((args.bind_host, args.udp_port))
        self.socket.setblocking(False)
        self.timer = self.create_timer(1.0 / args.poll_rate_hz, self.poll_once)
        self.get_logger().info(
            f"Listening on udp://{args.bind_host}:{args.udp_port} and forwarding to {args.cmd_vel_topic}"
        )

    def poll_once(self) -> None:
        while True:
            try:
                packet, _ = self.socket.recvfrom(65535)
            except BlockingIOError:
                break
            payload = self.parse_packet(packet)
            if payload is not None:
                self.publish_cmd_vel(*payload)

    def parse_packet(self, packet: bytes) -> tuple[float, float, float] | None:
        try:
            message = json.loads(packet.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            self.get_logger().warning(f"Ignoring malformed UDP packet: {exc}")
            return None

        try:
            linear_x = float(message.get("linear.x", 0.0))
            linear_y = float(message.get("linear.y", 0.0))
            angular_z = float(message.get("angular.z", 0.0))
        except (TypeError, ValueError) as exc:
            self.get_logger().warning(f"Ignoring invalid cmd_vel payload: {exc}")
            return None

        if self.args.enforce_limits:
            linear_x = clamp(linear_x, -self.args.max_linear_x, self.args.max_linear_x)
            linear_y = clamp(linear_y, -self.args.max_linear_y, self.args.max_linear_y)
            angular_z = clamp(angular_z, -self.args.max_angular_z, self.args.max_angular_z)

        return linear_x, linear_y, angular_z

    def publish_cmd_vel(self, linear_x: float, linear_y: float, angular_z: float) -> None:
        twist = Twist()
        twist.linear.x = linear_x
        twist.linear.y = linear_y
        twist.angular.z = angular_z
        self.publisher.publish(twist)
        if self.args.verbose:
            self.get_logger().info(
                f"Published /cmd_vel: linear.x={linear_x:.3f}, linear.y={linear_y:.3f}, angular.z={angular_z:.3f}"
            )

    def destroy_node(self) -> bool:
        self.socket.close()
        return super().destroy_node()


def main() -> None:
    args = parse_args()
    rclpy.init()
    node = UdpCmdVelBridge(args)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

"""Send a fixed cmd_vel command over UDP to the ROS2 bridge."""

from __future__ import annotations

import argparse
import json
import socket
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send cmd_vel over UDP")
    parser.add_argument("--udp-host", default="127.0.0.1")
    parser.add_argument("--udp-port", type=int, default=5555)
    parser.add_argument("--linear-x", type=float, default=0.0)
    parser.add_argument("--linear-y", type=float, default=0.0)
    parser.add_argument("--angular-z", type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=2.0, help="Seconds to stream the command")
    parser.add_argument("--rate-hz", type=float, default=20.0, help="How often to send the command")
    parser.add_argument("--stop-repeat", type=int, default=3, help="How many zero-velocity packets to send at the end")
    return parser.parse_args()


def send_packet(sock: socket.socket, host: str, port: int, linear_x: float, linear_y: float, angular_z: float) -> None:
    payload = {
        "source": "manual_cmd_vel_udp",
        "timestamp": time.time(),
        "linear.x": linear_x,
        "linear.y": linear_y,
        "angular.z": angular_z,
    }
    sock.sendto(json.dumps(payload).encode("utf-8"), (host, port))


def main() -> None:
    args = parse_args()
    interval = 1.0 / args.rate_hz
    end_time = time.monotonic() + max(0.0, args.duration)

    print(
        "Sending cmd_vel:",
        {
            "linear.x": args.linear_x,
            "linear.y": args.linear_y,
            "angular.z": args.angular_z,
            "duration": args.duration,
            "rate_hz": args.rate_hz,
            "udp": f"{args.udp_host}:{args.udp_port}",
        },
    )

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        while time.monotonic() < end_time:
            send_packet(
                sock,
                args.udp_host,
                args.udp_port,
                args.linear_x,
                args.linear_y,
                args.angular_z,
            )
            time.sleep(interval)

        for _ in range(max(0, args.stop_repeat)):
            send_packet(sock, args.udp_host, args.udp_port, 0.0, 0.0, 0.0)
            time.sleep(interval)
    finally:
        sock.close()

    print("Done. Sent zero cmd_vel packets to stop the robot.")


if __name__ == "__main__":
    main()

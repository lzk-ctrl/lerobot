#!/usr/bin/env python

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from types import ModuleType

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

REPO_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_OBSERVATION_JSON = REPO_ROOT / "observation_with_image.json"


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Call call_pi05_server.py continuously at a fixed start-to-start request rate. "
            "Unknown arguments are forwarded to call_pi05_server.py."
        ),
        allow_abbrev=False,
    )
    parser.add_argument("--rate-hz", type=float, default=3.0, help="Request loop rate in Hz.")
    parser.add_argument("--max-cycles", type=int, default=0, help="0 means run forever.")
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Exit on the first failed cycle. By default failures are printed and the loop continues.",
    )
    parser.add_argument("--server-host", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=6001)
    parser.add_argument("--observation-json", default=str(DEFAULT_OBSERVATION_JSON))
    parser.add_argument("--send-mode", choices=["first", "chunk"], default="first")
    parser.add_argument("--actions-per-chunk", type=int, default=1)
    parser.add_argument(
        "--execution-mode",
        choices=["continuous_control", "task_execution"],
        default="continuous_control",
    )
    parser.add_argument("--cmd-vel-output", choices=["print", "udp", "print+udp"], default="print")

    args, forwarded_args = parser.parse_known_args()
    if forwarded_args and forwarded_args[0] == "--":
        forwarded_args = forwarded_args[1:]
    return args, forwarded_args


def build_pi05_client_args(
    args: argparse.Namespace,
    forwarded_args: list[str],
    call_pi05_server: ModuleType,
) -> argparse.Namespace:
    client_argv = [
        "--server-host",
        args.server_host,
        "--server-port",
        str(args.server_port),
        "--observation-json",
        args.observation_json,
        "--send-mode",
        args.send_mode,
        "--actions-per-chunk",
        str(args.actions_per_chunk),
        "--execution-mode",
        args.execution_mode,
        "--cmd-vel-output",
        args.cmd_vel_output,
        *forwarded_args,
    ]

    old_argv = sys.argv
    try:
        sys.argv = [str(SCRIPT_DIR / "call_pi05_server.py"), *client_argv]
        client_args = call_pi05_server.parse_args()
    finally:
        sys.argv = old_argv

    client_args.continuous = False
    return client_args


def sleep_until(target_time: float) -> None:
    remaining = target_time - time.perf_counter()
    if remaining > 0.0:
        time.sleep(remaining)


def main() -> None:
    args, forwarded_args = parse_args()
    if args.rate_hz <= 0.0:
        raise ValueError("--rate-hz must be > 0")
    if args.max_cycles < 0:
        raise ValueError("--max-cycles must be >= 0")

    import call_pi05_server

    client_args = build_pi05_client_args(args, forwarded_args, call_pi05_server)
    period_sec = 1.0 / args.rate_hz
    cycle_index = 0
    next_cycle_start = time.perf_counter()

    print(
        "fixed_rate_client:",
        f"rate_hz={args.rate_hz:.3f}",
        f"period_sec={period_sec:.6f}",
        f"server={client_args.server_host}:{client_args.server_port}",
        f"observation_json={client_args.observation_json}",
        flush=True,
    )

    while args.max_cycles == 0 or cycle_index < args.max_cycles:
        sleep_until(next_cycle_start)
        cycle_start = time.perf_counter()
        cycle_index += 1
        client_args._current_cycle_index = cycle_index

        print(f"fixed_rate_cycle: {cycle_index}", flush=True)
        try:
            call_pi05_server.execute_cycle(client_args)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            print(f"cycle_error: {exc}", file=sys.stderr, flush=True)
            if args.stop_on_error:
                raise

        elapsed_sec = time.perf_counter() - cycle_start
        next_cycle_start = cycle_start + period_sec
        if elapsed_sec > period_sec:
            print(
                "rate_warning:",
                f"cycle_time_sec={elapsed_sec:.6f}",
                f"> period_sec={period_sec:.6f}; next cycle starts immediately",
                file=sys.stderr,
                flush=True,
            )
            next_cycle_start = time.perf_counter()


if __name__ == "__main__":
    main()

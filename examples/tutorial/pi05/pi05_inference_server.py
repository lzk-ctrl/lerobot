#!/usr/bin/env python

from __future__ import annotations

import argparse
import socket
import socketserver
import sys
import threading
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from lerobot.utils.nvtx import nvtx_range

from pi05_runtime import PI05InferenceEngine, recv_message, send_message  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Persistent PI0.5 inference server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6001)
    parser.add_argument("--model-id", default="lerobot/pi05_base", help="HF model id or local path")
    parser.add_argument(
        "--device",
        default="auto",
        help="Target inference device, e.g. auto, cpu, mps, cuda, cuda:1",
    )
    parser.add_argument("--state-dim", type=int, default=22)
    parser.add_argument("--action-dim", type=int, default=3)
    parser.add_argument("--image-height", type=int, default=480)
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50,
        help="Internal action chunk size/action horizon used by the PI0.5 model.",
    )
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument(
        "--default-num-action-samples",
        type=int,
        default=1,
        help="Default number of noise samples/action chunks to infer per observation.",
    )
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument(
        "--tokenizer-path",
        default=None,
        help="Optional local tokenizer directory. Defaults to <model-id>/tokenizer when present.",
    )
    parser.add_argument(
        "--denoising-debug-dir",
        default=None,
        help=(
            "Optional output directory for per-request denoising debug dumps. "
            "When set, the server saves noise/x_t/v_t/delta_t tensors and per-step summaries."
        ),
    )
    parser.add_argument(
        "--reusable-noise-path",
        default=None,
        help=(
            "Optional .pt path for warm-start noise reuse. "
            "If the file exists, the server loads it as the initial noise. "
            "If it does not exist, the server samples random noise once and then saves the final denoised result "
            "to this path for the next request."
        ),
    )
    parser.add_argument("--default-actions-per-chunk", type=int, default=1)
    parser.add_argument(
        "--max-infer-requests",
        type=int,
        default=0,
        help="Exit the server after this many successful infer requests. 0 means no limit.",
    )
    return parser.parse_args()


class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(
        self,
        server_address,
        request_handler_class,
        engine,
        default_actions_per_chunk,
        default_num_action_samples,
        max_infer_requests,
    ):
        super().__init__(server_address, request_handler_class)
        self.engine = engine
        self.default_actions_per_chunk = default_actions_per_chunk
        self.default_num_action_samples = default_num_action_samples
        self.max_infer_requests = max_infer_requests
        self._infer_request_count = 0
        self._infer_request_lock = threading.Lock()

    def record_infer_request_and_should_shutdown(self) -> bool:
        if self.max_infer_requests <= 0:
            return False
        with self._infer_request_lock:
            self._infer_request_count += 1
            return self._infer_request_count >= self.max_infer_requests


class PI05RequestHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        sock: socket.socket = self.request
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        while True:
            try:
                request = recv_message(sock)
            except EOFError:
                return
            except Exception as exc:
                send_message(sock, {"ok": False, "error": f"Failed to parse request: {exc}"})
                return

            request_type = request.get("type", "infer")
            if request_type == "ping":
                send_message(
                    sock,
                    {
                        "ok": True,
                        "message": "pong",
                        "timestamp": time.time(),
                    },
                )
                continue

            if request_type != "infer":
                send_message(sock, {"ok": False, "error": f"Unsupported request type: {request_type}"})
                continue

            try:
                with nvtx_range("pi05.server.handle_infer"):
                    task = request.get("task", "")
                    observation = request["observation"]
                    actions_per_chunk = int(
                        request.get("actions_per_chunk", self.server.default_actions_per_chunk)
                    )
                    num_action_samples = int(
                        request.get("num_action_samples", self.server.default_num_action_samples)
                    )
                    result = self.server.engine.infer(
                        observation=observation,
                        task=task,
                        actions_per_chunk=actions_per_chunk,
                        num_action_samples=num_action_samples,
                        initial_noise_path=request.get("initial_noise_path"),
                        save_final_noise_path=request.get("save_final_noise_path"),
                    )
                    send_message(sock, {"ok": True, **result})
                    if self.server.record_infer_request_and_should_shutdown():
                        threading.Thread(target=self.server.shutdown, daemon=True).start()
            except Exception as exc:
                send_message(sock, {"ok": False, "error": str(exc)})


def main() -> None:
    args = parse_args()
    engine = PI05InferenceEngine(
        model_id=args.model_id,
        device=args.device,
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        image_height=args.image_height,
        image_width=args.image_width,
        chunk_size=args.chunk_size,
        num_inference_steps=args.num_inference_steps,
        num_action_samples=args.default_num_action_samples,
        compile_model=args.compile_model,
        tokenizer_path=args.tokenizer_path,
        denoising_debug_dir=args.denoising_debug_dir,
        reusable_noise_path=args.reusable_noise_path,
    )

    server = ThreadedTCPServer(
        (args.host, args.port),
        PI05RequestHandler,
        engine=engine,
        default_actions_per_chunk=args.default_actions_per_chunk,
        default_num_action_samples=args.default_num_action_samples,
        max_infer_requests=args.max_infer_requests,
    )
    print(f"PI05 server listening on {args.host}:{args.port}", flush=True)
    print(f"default_num_action_samples: {args.default_num_action_samples}", flush=True)
    if args.denoising_debug_dir:
        print(f"Denoising debug dumps will be saved to {Path(args.denoising_debug_dir).expanduser()}", flush=True)
    if args.reusable_noise_path:
        print(f"Reusable noise state will be read/written at {Path(args.reusable_noise_path).expanduser()}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

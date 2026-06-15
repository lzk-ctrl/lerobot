#!/usr/bin/env python

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_STEPS = [4, 6, 8, 10]
DEFAULT_TOTAL_MS = [162.17, 219.30, 272.11, 327.10]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot PI0.5 VLA latency vs denoising steps")
    parser.add_argument(
        "--csv",
        default=None,
        help="Optional CSV path with columns num_inference_steps and total_ms or predict_action_chunk_ms.",
    )
    parser.add_argument(
        "--y-column",
        default="total_ms",
        choices=["total_ms", "predict_action_chunk_ms"],
        help="Y-axis column to read from CSV.",
    )
    parser.add_argument(
        "--steps",
        default="4,6,8,10",
        help="Comma-separated denoising step counts. Used when --csv is not provided.",
    )
    parser.add_argument(
        "--latencies-ms",
        default="162.17,219.30,272.11,327.10",
        help="Comma-separated latency values in milliseconds. Used when --csv is not provided.",
    )
    parser.add_argument(
        "--output",
        default="/tmp/pi05_latency_vs_steps.png",
        help="Output image path.",
    )
    parser.add_argument(
        "--title",
        default="PI0.5 VLA Inference Latency vs Denoising Steps",
        help="Plot title.",
    )
    parser.add_argument(
        "--subtitle",
        default="Latency grows approximately linearly with denoising steps",
        help="Subtitle shown below the title.",
    )
    parser.add_argument("--show", action="store_true", help="Display the figure after saving.")
    return parser.parse_args()


def parse_int_list(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("No valid step values were provided.")
    return values


def parse_float_list(raw: str) -> list[float]:
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("No valid latency values were provided.")
    return values


def load_from_csv(csv_path: str, y_column: str) -> tuple[list[int], list[float]]:
    steps: list[int] = []
    latencies_ms: list[float] = []

    with Path(csv_path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required_columns = {"num_inference_steps", y_column}
        if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
            raise ValueError(
                f"CSV must contain columns {sorted(required_columns)}, got {reader.fieldnames or []}"
            )

        for row in reader:
            steps.append(int(row["num_inference_steps"]))
            latencies_ms.append(float(row[y_column]))

    if not steps:
        raise ValueError(f"No rows found in CSV: {csv_path}")

    sorted_pairs = sorted(zip(steps, latencies_ms, strict=True))
    sorted_steps, sorted_latencies_ms = zip(*sorted_pairs, strict=True)
    return list(sorted_steps), list(sorted_latencies_ms)


def load_data(args: argparse.Namespace) -> tuple[list[int], list[float]]:
    if args.csv:
        return load_from_csv(args.csv, args.y_column)

    steps = parse_int_list(args.steps)
    latencies_ms = parse_float_list(args.latencies_ms)
    if len(steps) != len(latencies_ms):
        raise ValueError(
            f"--steps and --latencies-ms must have the same length, got {len(steps)} and {len(latencies_ms)}"
        )
    return steps, latencies_ms


def fit_line(steps: list[int], latencies_ms: list[float]) -> tuple[np.ndarray, float, float]:
    x = np.array(steps, dtype=np.float64)
    y = np.array(latencies_ms, dtype=np.float64)
    slope, intercept = np.polyfit(x, y, deg=1)
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 1.0
    return y_pred, float(slope), float(r_squared)


def plot_latency_vs_steps(
    steps: list[int],
    latencies_ms: list[float],
    output_path: str,
    title: str,
    subtitle: str,
    show: bool,
) -> None:
    y_fit, slope, r_squared = fit_line(steps, latencies_ms)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=160)

    ax.plot(
        steps,
        latencies_ms,
        color="#1f77b4",
        linewidth=2.2,
        marker="o",
        markersize=7,
        label="Measured latency",
    )
    ax.plot(
        steps,
        y_fit,
        color="#d62728",
        linewidth=1.8,
        linestyle="--",
        label="Linear fit",
    )

    for step, latency in zip(steps, latencies_ms, strict=True):
        ax.annotate(
            f"{latency:.1f} ms",
            xy=(step, latency),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )

    ax.set_title(title, fontsize=15, fontweight="bold", pad=20)
    fig.text(0.125, 0.92, subtitle, fontsize=10, color="#4c566a")
    ax.set_xlabel("Denoising steps", fontsize=11)
    ax.set_ylabel("Inference latency (ms)", fontsize=11)
    ax.set_xticks(steps)

    text = f"Linear slope: {slope:.1f} ms/step\nR^2: {r_squared:.3f}"
    ax.text(
        0.98,
        0.04,
        text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#ffffff", "edgecolor": "#cbd5e1"},
    )

    ax.legend(loc="upper left")
    fig.tight_layout()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    print(f"saved_plot: {output}")
    print(f"linear_slope_ms_per_step: {slope:.3f}")
    print(f"r_squared: {r_squared:.4f}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    args = parse_args()
    steps, latencies_ms = load_data(args)
    plot_latency_vs_steps(
        steps=steps,
        latencies_ms=latencies_ms,
        output_path=args.output,
        title=args.title,
        subtitle=args.subtitle,
        show=args.show,
    )


if __name__ == "__main__":
    main()

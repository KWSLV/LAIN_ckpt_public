#!/usr/bin/env python3
"""Plot per-epoch Gate/Text/Object conditional mAP contributions."""

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


MODULES = (
    ("Scene Gate", "gate", "#1f77b4"),
    ("Text Adapter", "text", "#ff7f0e"),
    ("Object Adapter", "object", "#2ca02c"),
)


def contribution(row, module, metric):
    if module == "gate":
        value = row.get("gap", {}).get(metric)
    else:
        key = "text_adapter" if module == "text" else "object_conditioned_adapter"
        value = (
            row.get("adapter_ablation", {})
            .get("contribution", {})
            .get(key, {})
            .get(metric)
        )
    return float(value) if value is not None else math.nan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not args.json_path.is_file():
        raise SystemExit(f"Diagnostics JSON not found: {args.json_path}")
    with args.json_path.open(encoding="utf-8") as handle:
        rows = json.load(handle)
    rows = sorted(rows, key=lambda row: int(row.get("epoch", 0)))
    if not rows:
        raise SystemExit(f"Diagnostics JSON is empty: {args.json_path}")

    epochs = [int(row.get("epoch", 0)) for row in rows]
    output = args.output or args.json_path.with_name("module_contributions.png")
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True)
    for axis, metric in zip(axes, ("full", "unseen", "seen")):
        for label, module, color in MODULES:
            values = [contribution(row, module, metric) for row in rows]
            axis.plot(epochs, values, marker="o", linewidth=2, label=label, color=color)
        axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
        axis.set_title(f"{metric.capitalize()} mAP contribution")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Full model - module OFF (mAP points)")
        axis.grid(True, linestyle="--", alpha=0.3)
        axis.set_xticks(epochs)
    axes[0].legend(loc="best")
    fig.suptitle("Per-epoch conditional module contributions")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[Contribution Plot] saved: {output}")


if __name__ == "__main__":
    main()

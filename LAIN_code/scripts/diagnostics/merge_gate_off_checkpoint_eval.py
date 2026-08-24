#!/usr/bin/env python3
"""Atomically merge one post-hoc Gate-OFF checkpoint result into diagnostics."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


METRICS = ("full", "unseen", "seen")


def numeric_summary(value, label):
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    summary = {}
    for metric in METRICS:
        number = value.get(metric)
        if not isinstance(number, (int, float)):
            raise ValueError(f"{label}.{metric} must be numeric")
        summary[metric] = float(number)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostics", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--expected-checkpoint", required=True, type=Path)
    args = parser.parse_args()

    result = json.loads(args.result.read_text(encoding="utf-8"))
    if result.get("evaluation") != "scene_gate_off_only":
        raise ValueError("Result is not a scene_gate_off_only evaluation")

    expected_checkpoint = args.expected_checkpoint.resolve()
    result_checkpoint = Path(result.get("checkpoint", "")).resolve()
    if result_checkpoint != expected_checkpoint:
        raise ValueError(
            f"Checkpoint mismatch: result={result_checkpoint}, expected={expected_checkpoint}"
        )
    if result.get("scene_gate_enabled_during_evaluation") is not False:
        raise ValueError("Result does not prove SceneGate was disabled")

    epoch = int(result["checkpoint_epoch"])
    iteration = int(result.get("checkpoint_iteration", 0))
    gate_off = numeric_summary(result.get("metrics"), "metrics")

    rows = json.loads(args.diagnostics.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("Diagnostics JSON must be a list")
    matching = [row for row in rows if int(row.get("epoch", -1)) == epoch]
    if len(matching) != 1:
        raise ValueError(f"Expected exactly one diagnostics row for epoch {epoch}")
    row = matching[0]
    gate_on = numeric_summary(row.get("on"), "on")
    contribution = {
        metric: gate_on[metric] - gate_off[metric]
        for metric in METRICS
    }

    training_off = row.get("off") if isinstance(row.get("off"), dict) else {}
    training_off_numeric = {}
    if all(isinstance(training_off.get(metric), (int, float)) for metric in METRICS):
        training_off_numeric = numeric_summary(training_off, "off")
    delta_vs_training_off = {
        metric: gate_off[metric] - training_off_numeric[metric]
        for metric in METRICS
    } if training_off_numeric else {}

    row["gate_ablation"] = {
        "definition": "scene_gate_contribution = full_model_gate_on - same_checkpoint_gate_off",
        "evaluation_mode": "posthoc_checkpoint_scene_gate_off_only",
        "checkpoint": expected_checkpoint.name,
        "checkpoint_epoch": epoch,
        "checkpoint_iteration": iteration,
        "evaluated_at": result.get("evaluated_at"),
        "gate_on": gate_on,
        "gate_off": gate_off,
        "contribution": contribution,
        "training_epoch_gate_off_before_posthoc": training_off_numeric,
        "posthoc_minus_training_epoch_gate_off": delta_vs_training_off,
        "scene_gate_enabled_during_evaluation": False,
    }

    # Maintain the historical top-level schema used by existing plotting and
    # analysis scripts, now backed by the independent checkpoint re-evaluation.
    row["off"] = gate_off
    row["gap"] = contribution
    rows.sort(key=lambda item: int(item.get("epoch", 0)))

    temporary = args.diagnostics.with_suffix(args.diagnostics.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, ensure_ascii=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, args.diagnostics)
    print(
        json.dumps(
            {
                "epoch": epoch,
                "checkpoint": expected_checkpoint.name,
                "gate_off": gate_off,
                "contribution": contribution,
                "diagnostics": str(args.diagnostics),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

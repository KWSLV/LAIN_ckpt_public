#!/usr/bin/env python3
"""Mirror one staged-retry experiment into a clean, continuous W&B run.

The training process remains untouched. Accepted epoch records are sourced from
scene_gate_diagnostics.json, while live iteration/LR status and console output
are mirrored from the authoritative training log.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import wandb


HEARTBEAT_RE = re.compile(
    r"\[W&B Heartbeat\]\s+logical_epoch=(\d+)\s+iteration=(\d+)\s+object_lr=([0-9.eE+-]+)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--train-log", required=True)
    parser.add_argument("--state-file", required=True)
    parser.add_argument("--entity", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--source-run", default="")
    parser.add_argument("--source-checkpoint", default="")
    parser.add_argument("--target-epochs", type=int, default=20)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument(
        "--resume-marker",
        default="[Resume] Completed epochs: 5; running",
        help="The last occurrence becomes the beginning of the mirrored console log.",
    )
    return parser.parse_args()


def atomic_json(path: Path, default: Any) -> Any:
    for _ in range(5):
        try:
            with path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except (FileNotFoundError, json.JSONDecodeError):
            time.sleep(0.2)
    return default


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def console_start_offset(log_path: Path, marker: str) -> int:
    try:
        data = log_path.read_bytes()
    except FileNotFoundError:
        return 0
    marker_bytes = marker.encode("utf-8")
    index = data.rfind(marker_bytes)
    if index < 0:
        return len(data)
    line_start = data.rfind(b"\n", 0, index)
    return 0 if line_start < 0 else line_start + 1


def finite_number(value: Any) -> float | int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return value
    return None


def accepted_epoch_metrics(record: dict[str, Any]) -> dict[str, Any]:
    epoch = int(record["epoch"])
    on = record.get("on") or {}
    gate_off = record.get("off") or on
    adapter = record.get("adapter_ablation") or {}
    adapter_off = adapter.get("off") or {}
    text_off = adapter_off.get("text_adapter") or on
    object_off = adapter_off.get("object_conditioned_adapter") or on
    staged = record.get("staged_training_metrics") or {}

    metrics: dict[str, Any] = {
        "logical_epoch": epoch,
        "mAP": on.get("full"),
        "unseen": on.get("unseen"),
        "seen": on.get("seen"),
        "module_contribution/epoch": epoch,
        "module_ablation/full_full": on.get("full"),
        "module_ablation/full_unseen": on.get("unseen"),
        "module_ablation/full_seen": on.get("seen"),
        "module_ablation/gate_off_full": gate_off.get("full"),
        "module_ablation/gate_off_unseen": gate_off.get("unseen"),
        "module_ablation/gate_off_seen": gate_off.get("seen"),
        "module_ablation/text_off_full": text_off.get("full"),
        "module_ablation/text_off_unseen": text_off.get("unseen"),
        "module_ablation/text_off_seen": text_off.get("seen"),
        "module_ablation/object_off_full": object_off.get("full"),
        "module_ablation/object_off_unseen": object_off.get("unseen"),
        "module_ablation/object_off_seen": object_off.get("seen"),
        "lr/head": staged.get("lr_head_used"),
        "lr/vit": staged.get("lr_vit_used"),
        "lr/object": staged.get("lr_object_used"),
        "lr_next/head": staged.get("lr_head_next"),
        "lr_next/vit": staged.get("lr_vit_next"),
        "lr_next/object": staged.get("lr_object_next"),
        "retry_count": staged.get("retry_count", 0),
        "rollback_count": staged.get("rollback_count", 0),
        "rollback_count_total": staged.get("rollback_count_total", 0),
        "object_forward_active": int(bool(staged.get("object_forward_active", False))),
        "one_time_lr_drop_applied": int(
            bool(staged.get("one_time_lr_drop_applied", False))
        ),
        "attempt": staged.get("attempt", 1),
        "attempt_index": staged.get("attempt_index"),
        "iteration_end": staged.get("iteration_end"),
        "decline_from_best": staged.get("decline_from_best"),
        "allowed_decline": staged.get("allowed_decline"),
    }

    for module, off_values in (
        ("gate", gate_off),
        ("text", text_off),
        ("object", object_off),
    ):
        for split in ("full", "unseen", "seen"):
            full_value = finite_number(on.get(split))
            off_value = finite_number(off_values.get(split))
            metrics[f"module_contribution/{module}_{split}"] = (
                full_value - off_value
                if full_value is not None and off_value is not None
                else None
            )

    return {key: value for key, value in metrics.items() if value is not None}


def fetch_rare_metrics(
    source_path: str, epoch: int, full_map: float | None
) -> dict[str, float]:
    if not source_path or full_map is None:
        return {}
    try:
        source = wandb.Api(timeout=30).run(source_path)
        match: dict[str, float] = {}
        for row in source.scan_history(page_size=1000):
            if row.get("logical_epoch") != epoch or row.get("mAP") is None:
                continue
            if abs(float(row["mAP"]) - float(full_map)) > 1e-7:
                continue
            for key in ("rare", "non-rare"):
                if row.get(key) is not None:
                    match[key] = float(row[key])
        return match
    except Exception as exc:  # Auxiliary fields must never stop the mirror.
        print(f"[Mirror Warning] rare/non-rare lookup failed: {exc}", flush=True)
        return {}


def define_metrics(run: wandb.sdk.wandb_run.Run) -> None:
    run.define_metric("logical_epoch")
    for pattern in (
        "mAP",
        "unseen",
        "seen",
        "rare",
        "non-rare",
        "lr/*",
        "lr_next/*",
        "module_ablation/*",
        "module_contribution/*",
        "retry_count",
        "rollback_count",
        "rollback_count_total",
        "object_forward_active",
        "one_time_lr_drop_applied",
        "attempt",
        "attempt_index",
        "iteration_end",
        "decline_from_best",
        "allowed_decline",
    ):
        run.define_metric(pattern, step_metric="logical_epoch")
    run.define_metric("runtime/iteration")
    run.define_metric("runtime/*", step_metric="runtime/iteration")


def main() -> int:
    args = parse_args()
    checkpoint_dir = Path(args.checkpoint_dir)
    diagnostics_path = checkpoint_dir / "scene_gate_diagnostics.json"
    retry_state_path = checkpoint_dir / "staged_retry_state.json"
    log_path = Path(args.train_log)
    state_path = Path(args.state_file)

    state = atomic_json(state_path, {})
    fresh = not bool(state.get("run_id"))
    if fresh:
        state = {
            "run_id": wandb.util.generate_id(),
            "mirrored_epochs": [],
            "last_runtime_iteration": 0,
            "log_offset": console_start_offset(log_path, args.resume_marker),
        }
        save_state(state_path, state)

    wandb_dir = state_path.parent / "wandb"
    wandb_dir.mkdir(parents=True, exist_ok=True)
    run = wandb.init(
        entity=args.entity,
        project=args.project,
        id=state["run_id"],
        name=args.run_name,
        group="LAIN5_Object15_dynamic_retry20_clean",
        resume="allow" if fresh else "must",
        dir=str(wandb_dir),
        config={
            "mirror/source_run": args.source_run,
            "mirror/source_checkpoint": args.source_checkpoint,
            "mirror/checkpoint_dir": str(checkpoint_dir),
            "mirror/authoritative_metrics": str(diagnostics_path),
            "mirror/authoritative_console": str(log_path),
            "mirror/accepted_epochs_only": True,
            "epochs": args.target_epochs,
            "object_lr_resumed": 2e-4,
        },
        settings=wandb.Settings(
            console="wrap",
            heartbeat_seconds=30,
            x_stats_sampling_interval=60,
        ),
    )
    if run is None:
        raise RuntimeError("wandb.init returned None")
    define_metrics(run)

    print(f"[Clean Mirror] run_id={run.id}", flush=True)
    print(f"[Clean Mirror] url={run.url}", flush=True)
    print(
        f"[Clean Mirror] accepted metrics source={diagnostics_path}", flush=True
    )
    print(f"[Clean Mirror] console source={log_path}", flush=True)

    mirrored_epochs = {int(value) for value in state.get("mirrored_epochs", [])}
    last_runtime_iteration = int(state.get("last_runtime_iteration", 0))
    log_offset = int(state.get("log_offset", 0))

    while True:
        records = atomic_json(diagnostics_path, [])
        for record in sorted(records, key=lambda item: int(item.get("epoch", 0))):
            epoch = int(record.get("epoch", 0))
            staged = record.get("staged_training_metrics") or {}
            if epoch <= 0 or epoch in mirrored_epochs or staged.get("accepted") is False:
                continue
            metrics = accepted_epoch_metrics(record)
            metrics.update(fetch_rare_metrics(args.source_run, epoch, metrics.get("mAP")))
            run.log(metrics)
            mirrored_epochs.add(epoch)
            state["mirrored_epochs"] = sorted(mirrored_epochs)
            save_state(state_path, state)
            print(
                "[Clean Mirror] accepted epoch "
                f"{epoch}: full={metrics.get('mAP'):.6f}, "
                f"unseen={metrics.get('unseen'):.6f}, "
                f"seen={metrics.get('seen'):.6f}",
                flush=True,
            )

        try:
            with log_path.open("rb") as handle:
                handle.seek(log_offset)
                chunk = handle.read()
                log_offset = handle.tell()
        except FileNotFoundError:
            chunk = b""

        if chunk:
            text = chunk.decode("utf-8", errors="replace")
            sys.stdout.write(text)
            if not text.endswith("\n"):
                sys.stdout.write("\n")
            sys.stdout.flush()
            for match in HEARTBEAT_RE.finditer(text):
                epoch = int(match.group(1))
                iteration = int(match.group(2))
                object_lr = float(match.group(3))
                if iteration <= last_runtime_iteration:
                    continue
                retry_state = atomic_json(retry_state_path, {})
                current_lrs = retry_state.get("current_lrs") or {}
                run.log(
                    {
                        "runtime/iteration": iteration,
                        "runtime/logical_epoch_in_progress": epoch,
                        "runtime/lr_head": current_lrs.get("head"),
                        "runtime/lr_vit": current_lrs.get("vit"),
                        "runtime/lr_object": object_lr,
                        "runtime/retry_count": 0,
                        "runtime/heartbeat_unix": time.time(),
                    }
                )
                last_runtime_iteration = iteration

            state["last_runtime_iteration"] = last_runtime_iteration
            state["log_offset"] = log_offset
            save_state(state_path, state)

        if mirrored_epochs and max(mirrored_epochs) >= args.target_epochs:
            print(
                f"[Clean Mirror] target epoch {args.target_epochs} mirrored; finishing.",
                flush=True,
            )
            run.finish(exit_code=0)
            return 0

        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())

"""Evaluation-time seen/unseen confusion diagnostics for HICO-DET."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from collections import defaultdict


SEEN = 0
UNSEEN = 1
GROUP_NAMES = ("seen", "unseen")


def _as_int(value):
    if value is None:
        return None
    if hasattr(value, "item"):
        value = value.item()
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _safe_rate(numerator, denominator):
    return float(numerator) / float(denominator) if denominator else None


class HOIErrorAnalyzer:
    """Accumulate spatially matched Seen/Unseen confusion statistics.

    Confusion matrices use ground-truth groups as rows and predicted groups as
    columns, both ordered as ``[seen, unseen]``. Unmatched GT instances are
    deliberately kept outside the 2x2 classification matrix so localisation
    misses cannot be mistaken for Seen/Unseen classification bias.
    """

    def __init__(
        self,
        unseen_ids,
        seen_ids=None,
        num_classes=600,
        unseen_object_ids=None,
        seen_object_ids=None,
        num_object_classes=80,
        iou_threshold=0.5,
    ):
        self.unseen_ids = set(int(idx) for idx in unseen_ids)
        if seen_ids is None:
            seen_ids = set(range(num_classes)) - self.unseen_ids
        self.seen_ids = set(int(idx) for idx in seen_ids) - self.unseen_ids

        self.object_analysis_applicable = unseen_object_ids is not None
        self.unseen_object_ids = set(int(idx) for idx in (unseen_object_ids or []))
        if self.object_analysis_applicable:
            if seen_object_ids is None:
                seen_object_ids = (
                    set(range(num_object_classes)) - self.unseen_object_ids
                )
            self.seen_object_ids = (
                set(int(idx) for idx in seen_object_ids)
                - self.unseen_object_ids
            )
        else:
            self.seen_object_ids = set()

        self.iou_threshold = float(iou_threshold)
        self.hoi_matrix = [[0, 0], [0, 0]]
        self.hoi_unmatched = [0, 0]
        self.hoi_unknown_prediction = [0, 0]
        self.object_matrix = [[0, 0], [0, 0]]
        self.object_unmatched = [0, 0]
        self.object_unknown_prediction = [0, 0]
        self.top_seen_bias = defaultdict(int)
        self.top_unseen_reverse_bias = defaultdict(int)
        self.top_object_seen_bias = defaultdict(int)
        self.top_object_unseen_reverse_bias = defaultdict(int)

    @staticmethod
    def _group(class_id, unseen_ids, seen_ids):
        class_id = _as_int(class_id)
        if class_id in unseen_ids:
            return UNSEEN
        if class_id in seen_ids:
            return SEEN
        return None

    def _update_level(
        self,
        gt_ids,
        top_pred_ids,
        unseen_ids,
        seen_ids,
        matrix,
        unmatched,
        unknown_prediction,
        top_seen_bias,
        top_unseen_reverse_bias,
    ):
        if len(gt_ids) != len(top_pred_ids):
            raise ValueError("gt_ids and top_pred_ids must have the same length")

        for gt_id, top_pred_id in zip(gt_ids, top_pred_ids):
            gt_id = _as_int(gt_id)
            actual_group = self._group(gt_id, unseen_ids, seen_ids)
            if actual_group is None:
                continue

            if top_pred_id is None:
                unmatched[actual_group] += 1
                continue

            top_pred_id = _as_int(top_pred_id)
            predicted_group = self._group(top_pred_id, unseen_ids, seen_ids)
            if predicted_group is None:
                unknown_prediction[actual_group] += 1
                continue

            matrix[actual_group][predicted_group] += 1
            if actual_group == UNSEEN and predicted_group == SEEN:
                top_seen_bias[top_pred_id] += 1
            elif actual_group == SEEN and predicted_group == UNSEEN:
                top_unseen_reverse_bias[top_pred_id] += 1

    def update(self, gt_hois, top_pred_id):
        """Backward-compatible wrapper for a shared top prediction."""
        self.update_aligned(gt_hois, [top_pred_id] * len(gt_hois))

    def update_aligned(self, gt_hois, top_pred_ids):
        self._update_level(
            gt_hois,
            top_pred_ids,
            self.unseen_ids,
            self.seen_ids,
            self.hoi_matrix,
            self.hoi_unmatched,
            self.hoi_unknown_prediction,
            self.top_seen_bias,
            self.top_unseen_reverse_bias,
        )

    def update_objects_aligned(self, gt_objects, top_pred_object_ids):
        if not self.object_analysis_applicable:
            return
        self._update_level(
            gt_objects,
            top_pred_object_ids,
            self.unseen_object_ids,
            self.seen_object_ids,
            self.object_matrix,
            self.object_unmatched,
            self.object_unknown_prediction,
            self.top_object_seen_bias,
            self.top_object_unseen_reverse_bias,
        )

    def merge(self, other):
        for row in range(2):
            for column in range(2):
                self.hoi_matrix[row][column] += other.hoi_matrix[row][column]
                self.object_matrix[row][column] += other.object_matrix[row][column]
            self.hoi_unmatched[row] += other.hoi_unmatched[row]
            self.hoi_unknown_prediction[row] += other.hoi_unknown_prediction[row]
            self.object_unmatched[row] += other.object_unmatched[row]
            self.object_unknown_prediction[row] += other.object_unknown_prediction[row]

        for destination, source in (
            (self.top_seen_bias, other.top_seen_bias),
            (self.top_unseen_reverse_bias, other.top_unseen_reverse_bias),
            (self.top_object_seen_bias, other.top_object_seen_bias),
            (
                self.top_object_unseen_reverse_bias,
                other.top_object_unseen_reverse_bias,
            ),
        ):
            for class_id, count in source.items():
                destination[int(class_id)] += int(count)

    def state_dict(self):
        return {
            "schema_version": 2,
            "iou_threshold": self.iou_threshold,
            "object_analysis_applicable": self.object_analysis_applicable,
            "hoi_matrix": self.hoi_matrix,
            "hoi_unmatched": self.hoi_unmatched,
            "hoi_unknown_prediction": self.hoi_unknown_prediction,
            "object_matrix": self.object_matrix,
            "object_unmatched": self.object_unmatched,
            "object_unknown_prediction": self.object_unknown_prediction,
            "top_seen_bias": dict(self.top_seen_bias),
            "top_unseen_reverse_bias": dict(self.top_unseen_reverse_bias),
            "top_object_seen_bias": dict(self.top_object_seen_bias),
            "top_object_unseen_reverse_bias": dict(
                self.top_object_unseen_reverse_bias
            ),
        }

    def load_state_dict(self, state):
        if int(state.get("schema_version", 1)) < 2:
            total_unseen = int(state.get("total_unseen", 0))
            bias_to_seen = int(state.get("bias_to_seen", 0))
            missed_unseen = int(state.get("missed_unseen", 0))
            self.hoi_matrix[UNSEEN][SEEN] = bias_to_seen
            self.hoi_unmatched[UNSEEN] = max(0, missed_unseen - bias_to_seen)
            self.hoi_matrix[UNSEEN][UNSEEN] = max(0, total_unseen - missed_unseen)
        else:
            self.iou_threshold = float(state.get("iou_threshold", self.iou_threshold))
            self.object_analysis_applicable = bool(
                state.get(
                    "object_analysis_applicable",
                    self.object_analysis_applicable,
                )
            )
            self.hoi_matrix = [list(map(int, row)) for row in state["hoi_matrix"]]
            self.hoi_unmatched = list(map(int, state["hoi_unmatched"]))
            self.hoi_unknown_prediction = list(
                map(int, state.get("hoi_unknown_prediction", [0, 0]))
            )
            self.object_matrix = [
                list(map(int, row))
                for row in state.get("object_matrix", [[0, 0], [0, 0]])
            ]
            self.object_unmatched = list(
                map(int, state.get("object_unmatched", [0, 0]))
            )
            self.object_unknown_prediction = list(
                map(int, state.get("object_unknown_prediction", [0, 0]))
            )

        for attribute in (
            "top_seen_bias",
            "top_unseen_reverse_bias",
            "top_object_seen_bias",
            "top_object_unseen_reverse_bias",
        ):
            setattr(
                self,
                attribute,
                defaultdict(
                    int,
                    {
                        int(key): int(value)
                        for key, value in state.get(attribute, {}).items()
                    },
                ),
            )

    @staticmethod
    def _level_summary(matrix, unmatched, unknown_prediction):
        seen_to_seen = int(matrix[SEEN][SEEN])
        seen_to_unseen = int(matrix[SEEN][UNSEEN])
        unseen_to_seen = int(matrix[UNSEEN][SEEN])
        unseen_to_unseen = int(matrix[UNSEEN][UNSEEN])
        matched_seen = seen_to_seen + seen_to_unseen
        matched_unseen = unseen_to_seen + unseen_to_unseen
        total_seen = matched_seen + int(unmatched[SEEN]) + int(unknown_prediction[SEEN])
        total_unseen = (
            matched_unseen
            + int(unmatched[UNSEEN])
            + int(unknown_prediction[UNSEEN])
        )
        return {
            "labels": list(GROUP_NAMES),
            "matrix_orientation": "rows=ground_truth, columns=prediction",
            "confusion_matrix_counts": [
                [seen_to_seen, seen_to_unseen],
                [unseen_to_seen, unseen_to_unseen],
            ],
            "total_seen_gt": total_seen,
            "total_unseen_gt": total_unseen,
            "matched_seen_gt": matched_seen,
            "matched_unseen_gt": matched_unseen,
            "unmatched_seen_gt": int(unmatched[SEEN]),
            "unmatched_unseen_gt": int(unmatched[UNSEEN]),
            "unknown_prediction_seen_gt": int(unknown_prediction[SEEN]),
            "unknown_prediction_unseen_gt": int(unknown_prediction[UNSEEN]),
            "seen_to_seen": seen_to_seen,
            "seen_to_unseen": seen_to_unseen,
            "unseen_to_seen": unseen_to_seen,
            "unseen_to_unseen": unseen_to_unseen,
            "rates": {
                "S2S_matched": _safe_rate(seen_to_seen, matched_seen),
                "S2U_matched": _safe_rate(seen_to_unseen, matched_seen),
                "S2U_all": _safe_rate(seen_to_unseen, total_seen),
                "seen_miss_rate": _safe_rate(unmatched[SEEN], total_seen),
                "U2S_matched": _safe_rate(unseen_to_seen, matched_unseen),
                "U2S_all": _safe_rate(unseen_to_seen, total_unseen),
                "U2U_matched": _safe_rate(unseen_to_unseen, matched_unseen),
                "unseen_miss_rate": _safe_rate(unmatched[UNSEEN], total_unseen),
            },
        }

    def summary(self):
        hoi = self._level_summary(
            self.hoi_matrix,
            self.hoi_unmatched,
            self.hoi_unknown_prediction,
        )
        hoi["top_unseen_to_seen_predictions"] = [
            {"predicted_hoi_id": class_id, "count": count}
            for class_id, count in sorted(
                self.top_seen_bias.items(),
                key=lambda item: (-item[1], item[0]),
            )
        ]
        hoi["top_seen_to_unseen_predictions"] = [
            {"predicted_hoi_id": class_id, "count": count}
            for class_id, count in sorted(
                self.top_unseen_reverse_bias.items(),
                key=lambda item: (-item[1], item[0]),
            )
        ]

        if self.object_analysis_applicable:
            object_level = self._level_summary(
                self.object_matrix,
                self.object_unmatched,
                self.object_unknown_prediction,
            )
            object_level["applicable"] = True
            object_level["top_unseen_to_seen_predictions"] = [
                {"predicted_object_id": class_id, "count": count}
                for class_id, count in sorted(
                    self.top_object_seen_bias.items(),
                    key=lambda item: (-item[1], item[0]),
                )
            ]
            object_level["top_seen_to_unseen_predictions"] = [
                {"predicted_object_id": class_id, "count": count}
                for class_id, count in sorted(
                    self.top_object_unseen_reverse_bias.items(),
                    key=lambda item: (-item[1], item[0]),
                )
            ]
        else:
            object_level = {
                "applicable": False,
                "reason": "Object-level Seen/Unseen groups are defined only for unseen_object.",
            }

        return {
            "schema_version": 2,
            "definition": (
                "A prediction is matched to a GT HOI when both human and object "
                f"IoU are >= {self.iou_threshold:g}; the highest-scoring matched "
                "HOI prediction determines the predicted Seen/Unseen group."
            ),
            "iou_threshold": self.iou_threshold,
            "hoi_level": hoi,
            "object_level": object_level,
        }

    def report(self):
        summary = self.summary()
        hoi = summary["hoi_level"]
        rates = hoi["rates"]
        print("\n" + "=" * 64)
        print("Seen/Unseen spatially matched classification confusion")
        print("Rows=GT, columns=prediction; labels=[Seen, Unseen]")
        print(f"HOI counts: {hoi['confusion_matrix_counts']}")
        print(
            "HOI U->S: "
            f"{hoi['unseen_to_seen']}/{hoi['matched_unseen_gt']} "
            f"({(rates['U2S_matched'] or 0.0) * 100:.2f}% matched); "
            f"unmatched unseen={hoi['unmatched_unseen_gt']}"
        )
        print(
            "HOI S->U: "
            f"{hoi['seen_to_unseen']}/{hoi['matched_seen_gt']} "
            f"({(rates['S2U_matched'] or 0.0) * 100:.2f}% matched); "
            f"unmatched seen={hoi['unmatched_seen_gt']}"
        )
        if summary["object_level"].get("applicable"):
            obj = summary["object_level"]
            obj_rates = obj["rates"]
            print(f"Object counts (unique GT pairs): {obj['confusion_matrix_counts']}")
            print(
                "Object U->S: "
                f"{obj['unseen_to_seen']}/{obj['matched_unseen_gt']} "
                f"({(obj_rates['U2S_matched'] or 0.0) * 100:.2f}% matched)"
            )
            print(
                "Object S->U: "
                f"{obj['seen_to_unseen']}/{obj['matched_seen_gt']} "
                f"({(obj_rates['S2U_matched'] or 0.0) * 100:.2f}% matched)"
            )
        print("=" * 64)

    @staticmethod
    def _write_csv(path, summary):
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                [
                    "level",
                    "actual_group",
                    "predicted_group",
                    "count",
                    "matched_row_total",
                    "rate_within_matched_row",
                ]
            )
            for level in ("hoi_level", "object_level"):
                data = summary[level]
                if level == "object_level" and not data.get("applicable"):
                    continue
                matrix = data["confusion_matrix_counts"]
                for row, actual in enumerate(GROUP_NAMES):
                    row_total = sum(matrix[row])
                    for column, predicted in enumerate(GROUP_NAMES):
                        count = matrix[row][column]
                        writer.writerow(
                            [
                                level,
                                actual,
                                predicted,
                                count,
                                row_total,
                                _safe_rate(count, row_total),
                            ]
                        )
                    writer.writerow(
                        [
                            level,
                            actual,
                            "unmatched",
                            data[f"unmatched_{actual}_gt"],
                            row_total,
                            "",
                        ]
                    )

    @staticmethod
    def _plot(path, title, data):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        counts = np.asarray(data["confusion_matrix_counts"], dtype=float)
        row_totals = counts.sum(axis=1, keepdims=True)
        percentages = np.divide(
            counts,
            row_totals,
            out=np.zeros_like(counts),
            where=row_totals != 0,
        ) * 100.0
        figure, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), constrained_layout=True)
        for axis, values, subtitle, suffix in (
            (axes[0], counts, "Counts", ""),
            (axes[1], percentages, "Row-normalized", "%"),
        ):
            image = axis.imshow(values, cmap="Blues", vmin=0)
            axis.set_xticks([0, 1], ["Seen", "Unseen"])
            axis.set_yticks([0, 1], ["Seen", "Unseen"])
            axis.set_xlabel("Predicted group")
            axis.set_ylabel("Ground-truth group")
            axis.set_title(subtitle)
            for row in range(2):
                for column in range(2):
                    label = (
                        f"{int(values[row, column])}"
                        if not suffix
                        else f"{values[row, column]:.2f}{suffix}"
                    )
                    axis.text(
                        column,
                        row,
                        label,
                        ha="center",
                        va="center",
                        color=("white" if values[row, column] > values.max() / 2 else "black"),
                    )
            figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        figure.suptitle(title)
        figure.savefig(path, dpi=180)
        plt.close(figure)

    def save(self, output_dir, metadata=None, make_plots=True):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        result = self.summary()
        result["evaluated_at"] = datetime.now(timezone.utc).isoformat()
        result["metadata"] = dict(metadata or {})

        json_path = output_dir / "seen_unseen_confusion.json"
        temporary_path = json_path.with_suffix(json_path.suffix + ".tmp")
        with temporary_path.open("w", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, ensure_ascii=False)
        temporary_path.replace(json_path)
        self._write_csv(output_dir / "seen_unseen_confusion.csv", result)

        if make_plots:
            self._plot(
                output_dir / "hoi_seen_unseen_confusion.png",
                "HOI-level Seen/Unseen confusion",
                result["hoi_level"],
            )
            if result["object_level"].get("applicable"):
                self._plot(
                    output_dir / "object_seen_unseen_confusion.png",
                    "Object-level Seen/Unseen confusion (unique GT pairs)",
                    result["object_level"],
                )
        return result

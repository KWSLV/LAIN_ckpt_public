from collections import defaultdict


class HOIErrorAnalyzer:
    """Spatially matched prediction diagnostics for unseen HOI ground truths."""

    def __init__(self, unseen_ids, seen_ids=None, num_classes=600):
        self.unseen_ids = set(int(idx) for idx in unseen_ids)
        if seen_ids is None:
            seen_ids = set(range(num_classes)) - self.unseen_ids
        self.seen_ids = set(int(idx) for idx in seen_ids) - self.unseen_ids
        self.total_unseen = 0
        self.missed_unseen = 0
        self.bias_to_seen = 0
        self.top_seen_bias = defaultdict(int)

    def update(self, gt_hois, top_pred_id):
        self.update_aligned(gt_hois, [top_pred_id] * len(gt_hois))

    def update_aligned(self, gt_hois, top_pred_ids):
        if len(gt_hois) != len(top_pred_ids):
            raise ValueError("gt_hois and top_pred_ids must have the same length")
        for gt_id, top_pred_id in zip(gt_hois, top_pred_ids):
            gt_id = int(gt_id.item() if hasattr(gt_id, "item") else gt_id)
            if gt_id not in self.unseen_ids:
                continue

            top_pred_id = None if top_pred_id is None else int(top_pred_id)

            self.total_unseen += 1
            if top_pred_id == gt_id:
                continue

            self.missed_unseen += 1
            if top_pred_id in self.seen_ids:
                self.bias_to_seen += 1
                self.top_seen_bias[top_pred_id] += 1

    def merge(self, other):
        self.total_unseen += other.total_unseen
        self.missed_unseen += other.missed_unseen
        self.bias_to_seen += other.bias_to_seen
        for hoi_id, count in other.top_seen_bias.items():
            self.top_seen_bias[int(hoi_id)] += int(count)

    def state_dict(self):
        return {
            "total_unseen": self.total_unseen,
            "missed_unseen": self.missed_unseen,
            "bias_to_seen": self.bias_to_seen,
            "top_seen_bias": dict(self.top_seen_bias),
        }

    def load_state_dict(self, state):
        self.total_unseen = int(state.get("total_unseen", 0))
        self.missed_unseen = int(state.get("missed_unseen", 0))
        self.bias_to_seen = int(state.get("bias_to_seen", 0))
        self.top_seen_bias = defaultdict(
            int,
            {int(key): int(value) for key, value in state.get("top_seen_bias", {}).items()},
        )

    def report(self):
        print("\n" + "=" * 50)
        print("原始模型输出诊断（无分数校准）")
        print("-" * 50)
        if self.total_unseen == 0:
            print("未检测到 Unseen ground-truth 样本。")
            print("=" * 50)
            return

        miss_ratio = self.missed_unseen / self.total_unseen * 100
        seen_bias_ratio = self.bias_to_seen / self.total_unseen * 100
        error_seen_ratio = self.bias_to_seen / max(self.missed_unseen, 1) * 100

        print(f"Unseen 样本总数: {self.total_unseen}")
        print(f"漏检/误检率: {miss_ratio:.2f}%")
        print(f"Seen 类偏见比例: {seen_bias_ratio:.2f}%")
        print(f"错误预测中 Seen 类占比: {error_seen_ratio:.2f}%")
        print("Top 3 Seen 干扰项:")
        sorted_bias = sorted(self.top_seen_bias.items(), key=lambda item: item[1], reverse=True)
        if not sorted_bias:
            print("  - 无")
        for hoi_id, count in sorted_bias[:3]:
            print(f"  - ID {hoi_id}: {count} 次")
        print("=" * 50)

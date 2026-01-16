import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class ResultsComparator:
    """
    Compare and visualize two correspondence evaluation results dictionaries
    (e.g., baseline vs fine-tuned) with the same plot types as ResultsAnalyzer.
    """

    def __init__(self, metrics_a, metrics_b, name_a="baseline", name_b="fine_tuned"):
        self.A = metrics_a
        self.B = metrics_b
        self.name_a = name_a
        self.name_b = name_b

    # --------- helpers ---------
    def _thresholds_union(self):
        ta = set(self.A["overall"].keys())
        tb = set(self.B["overall"].keys())
        return sorted(ta | tb)

    def _get_safe(self, dct, keys, default=np.nan):
        cur = dct
        for k in keys:
            if cur is None or k not in cur:
                return default
            cur = cur[k]
        return cur

    def _save_show(self, fig, save_path):
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    # --------- plots ---------
    def plot_pck_curve(self, save_path=None):
        """Overlay PCK curves across thresholds (overall)"""
        thresholds = self._thresholds_union()
        pck_a = [self.A["overall"].get(t, np.nan) for t in thresholds]
        pck_b = [self.B["overall"].get(t, np.nan) for t in thresholds]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(thresholds, pck_a, marker="o", linewidth=2, markersize=8, label=self.name_a)
        ax.plot(thresholds, pck_b, marker="o", linewidth=2, markersize=8, label=self.name_b)

        ax.set_xlabel("PCK Threshold (α)", fontsize=12)
        ax.set_ylabel("PCK (%)", fontsize=12)
        ax.set_title("PCK vs Threshold (Comparison)", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 100])
        ax.legend()

        # annotate delta at each threshold if both exist
        for t, a, b in zip(thresholds, pck_a, pck_b):
            if np.isfinite(a) and np.isfinite(b):
                ax.text(t, max(a, b) + 2, f"Δ{(b-a):+.1f}", ha="center", fontsize=9)

        self._save_show(fig, save_path)

    def plot_per_category(self, threshold=0.10, save_path=None, sort_by="delta"):
        """
        Side-by-side bars per category at a fixed threshold.
        sort_by: "delta" | "b" | "a" | "name"
        """
        cats = sorted(set(self.A["per_category"].keys()) | set(self.B["per_category"].keys()))
        a_vals = [self._get_safe(self.A, ["per_category", c, threshold]) for c in cats]
        b_vals = [self._get_safe(self.B, ["per_category", c, threshold]) for c in cats]
        deltas = [bv - av if np.isfinite(av) and np.isfinite(bv) else np.nan for av, bv in zip(a_vals, b_vals)]

        # sorting
        order = list(range(len(cats)))
        if sort_by == "delta":
            order.sort(key=lambda i: (np.nan_to_num(deltas[i], nan=-1e9)), reverse=True)
        elif sort_by == "b":
            order.sort(key=lambda i: (np.nan_to_num(b_vals[i], nan=-1e9)), reverse=True)
        elif sort_by == "a":
            order.sort(key=lambda i: (np.nan_to_num(a_vals[i], nan=-1e9)), reverse=True)
        elif sort_by == "name":
            order.sort(key=lambda i: cats[i])

        cats = [cats[i] for i in order]
        a_vals = [a_vals[i] for i in order]
        b_vals = [b_vals[i] for i in order]
        deltas = [deltas[i] for i in order]

        x = np.arange(len(cats))
        w = 0.38

        fig, ax = plt.subplots(figsize=(12, 6))
        bars_a = ax.bar(x - w/2, a_vals, width=w, alpha=0.8, label=self.name_a)
        bars_b = ax.bar(x + w/2, b_vals, width=w, alpha=0.8, label=self.name_b)

        ax.set_xlabel("Category", fontsize=12)
        ax.set_ylabel(f"PCK@{threshold:.2f} (%)", fontsize=12)
        ax.set_title(f"Per-Category Performance (PCK@{threshold:.2f}) مقارنة", fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(cats, rotation=45, ha="right")
        ax.set_ylim([0, 100])
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend()

        # delta labels above the higher bar
        for i, (ba, bb, d) in enumerate(zip(bars_a, bars_b, deltas)):
            if np.isfinite(d):
                top = max(ba.get_height(), bb.get_height())
                ax.text(i, top + 1, f"Δ{d:+.1f}", ha="center", va="bottom", fontsize=9)

        self._save_show(fig, save_path)

    def plot_keypoint_difficulty(self, threshold=0.10, top_n=15, save_path=None):
        """
        Compare hardest/easiest keypoints using delta (B-A).
        Shows 2 panels: hardest regressions and biggest improvements.
        """
        # collect union keypoints: label = "{category}-{kp_id}"
        all_items = []
        cats = set(self.A["per_keypoint"].keys()) | set(self.B["per_keypoint"].keys())
        for c in cats:
            kps = set(self.A["per_keypoint"].get(c, {}).keys()) | set(self.B["per_keypoint"].get(c, {}).keys())
            for kp in kps:
                a = self._get_safe(self.A, ["per_keypoint", c, kp, threshold])
                b = self._get_safe(self.B, ["per_keypoint", c, kp, threshold])
                if np.isfinite(a) and np.isfinite(b):
                    all_items.append((f"{c}-{kp}", a, b, b - a))

        if len(all_items) == 0:
            print("No overlapping keypoints found for comparison.")
            return

        all_items.sort(key=lambda x: x[3])  # by delta asc
        worst = all_items[:top_n]           # most negative
        best = all_items[-top_n:][::-1]     # most positive

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

        # worst regressions
        labels_w = [x[0] for x in worst]
        deltas_w = [x[3] for x in worst]
        ax1.barh(range(len(worst)), deltas_w, alpha=0.8)
        ax1.set_yticks(range(len(worst)))
        ax1.set_yticklabels(labels_w, fontsize=9)
        ax1.set_xlabel(f"ΔPCK@{threshold:.2f} (B - A)", fontsize=12)
        ax1.set_title(f"Top {top_n} Regressions (worst Δ)", fontsize=14, fontweight="bold")
        ax1.grid(True, axis="x", alpha=0.3)

        for i, v in enumerate(deltas_w):
            ax1.text(v + (0.5 if v >= 0 else -0.5), i, f"{v:+.1f}", va="center",
                     ha="left" if v >= 0 else "right", fontsize=9)

        # best improvements
        labels_b = [x[0] for x in best]
        deltas_b = [x[3] for x in best]
        ax2.barh(range(len(best)), deltas_b, alpha=0.8)
        ax2.set_yticks(range(len(best)))
        ax2.set_yticklabels(labels_b, fontsize=9)
        ax2.set_xlabel(f"ΔPCK@{threshold:.2f} (B - A)", fontsize=12)
        ax2.set_title(f"Top {top_n} Improvements (best Δ)", fontsize=14, fontweight="bold")
        ax2.grid(True, axis="x", alpha=0.3)

        for i, v in enumerate(deltas_b):
            ax2.text(v + (0.5 if v >= 0 else -0.5), i, f"{v:+.1f}", va="center",
                     ha="left" if v >= 0 else "right", fontsize=9)

        self._save_show(fig, save_path)

    def plot_image_difficulty_distribution(self, threshold=0.10, save_path=None, bins=30):
        """Compare distributions of per-image PCK scores (two hist overlays)"""
        a_scores = [self.A["per_image"][idx][threshold] for idx in self.A["per_image"].keys()
                    if threshold in self.A["per_image"][idx]]
        b_scores = [self.B["per_image"][idx][threshold] for idx in self.B["per_image"].keys()
                    if threshold in self.B["per_image"][idx]]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(a_scores, bins=bins, alpha=0.5, edgecolor="black", label=self.name_a)
        ax.hist(b_scores, bins=bins, alpha=0.5, edgecolor="black", label=self.name_b)

        if len(a_scores) > 0:
            ax.axvline(np.mean(a_scores), linestyle="--", linewidth=2, label=f"{self.name_a} mean: {np.mean(a_scores):.1f}%")
        if len(b_scores) > 0:
            ax.axvline(np.mean(b_scores), linestyle="--", linewidth=2, label=f"{self.name_b} mean: {np.mean(b_scores):.1f}%")

        ax.set_xlabel(f"PCK@{threshold:.2f} (%)", fontsize=12)
        ax.set_ylabel("Number of Image Pairs", fontsize=12)
        ax.set_title("Distribution of Per-Image PCK Scores (Comparison)", fontsize=14, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        self._save_show(fig, save_path)

    def plot_per_category_keypoints(self, category, threshold=0.10, save_path=None, sort_by="delta"):
        """Compare per-keypoint performance for a specific category (side-by-side bars)"""
        if category not in self.A["per_keypoint"] and category not in self.B["per_keypoint"]:
            print(f"Category '{category}' not found in results.")
            return

        kpA = self.A["per_keypoint"].get(category, {})
        kpB = self.B["per_keypoint"].get(category, {})
        kp_ids = sorted(set(kpA.keys()) | set(kpB.keys()))

        a_vals = [self._get_safe(self.A, ["per_keypoint", category, kp, threshold]) for kp in kp_ids]
        b_vals = [self._get_safe(self.B, ["per_keypoint", category, kp, threshold]) for kp in kp_ids]
        deltas = [bv - av if np.isfinite(av) and np.isfinite(bv) else np.nan for av, bv in zip(a_vals, b_vals)]

        order = list(range(len(kp_ids)))
        if sort_by == "delta":
            order.sort(key=lambda i: (np.nan_to_num(deltas[i], nan=-1e9)), reverse=True)
        elif sort_by == "b":
            order.sort(key=lambda i: (np.nan_to_num(b_vals[i], nan=-1e9)), reverse=True)
        elif sort_by == "a":
            order.sort(key=lambda i: (np.nan_to_num(a_vals[i], nan=-1e9)), reverse=True)

        kp_ids = [kp_ids[i] for i in order]
        a_vals = [a_vals[i] for i in order]
        b_vals = [b_vals[i] for i in order]
        deltas = [deltas[i] for i in order]

        x = np.arange(len(kp_ids))
        w = 0.38

        fig, ax = plt.subplots(figsize=(12, 6))
        bars_a = ax.bar(x - w/2, a_vals, width=w, alpha=0.8, label=self.name_a)
        bars_b = ax.bar(x + w/2, b_vals, width=w, alpha=0.8, label=self.name_b)

        ax.set_xlabel("Keypoint ID", fontsize=12)
        ax.set_ylabel(f"PCK@{threshold:.2f} (%)", fontsize=12)
        ax.set_title(f"Per-Keypoint Performance for {category.capitalize()} (Comparison)",
                     fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(kp_ids, rotation=0)
        ax.set_ylim([0, 100])
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend()

        for i, (ba, bb, d) in enumerate(zip(bars_a, bars_b, deltas)):
            if np.isfinite(d):
                top = max(ba.get_height(), bb.get_height())
                ax.text(i, top + 1, f"Δ{d:+.1f}", ha="center", va="bottom", fontsize=8)

        self._save_show(fig, save_path)

    # --------- tables & export ---------
    def create_summary_table(self, threshold=0.10):
        """DataFrame with overall + per-category for A, B and delta (B-A)"""
        data = {
            "Overall A": [self.A["overall"].get(threshold, np.nan)],
            "Overall B": [self.B["overall"].get(threshold, np.nan)],
            "Δ Overall": [self.B["overall"].get(threshold, np.nan) - self.A["overall"].get(threshold, np.nan)],
        }

        cats = sorted(set(self.A["per_category"].keys()) | set(self.B["per_category"].keys()))
        for c in cats:
            a = self._get_safe(self.A, ["per_category", c, threshold])
            b = self._get_safe(self.B, ["per_category", c, threshold])
            data[f"{c} A"] = [a]
            data[f"{c} B"] = [b]
            data[f"{c} Δ"] = [b - a if np.isfinite(a) and np.isfinite(b) else np.nan]

        return pd.DataFrame(data)

    def export_to_csv(self, save_dir="./results"):
        """Export comparison CSVs (overall, per-category, per-keypoint, per-image)"""
        os.makedirs(save_dir, exist_ok=True)
        thresholds = self._thresholds_union()

        # overall
        overall_rows = []
        for t in thresholds:
            a = self.A["overall"].get(t, np.nan)
            b = self.B["overall"].get(t, np.nan)
            overall_rows.append({"threshold": t, f"pck_{self.name_a}": a, f"pck_{self.name_b}": b, "delta": b - a})
        pd.DataFrame(overall_rows).to_csv(os.path.join(save_dir, "overall_pck_compare.csv"), index=False)

        # per-category
        cats = sorted(set(self.A["per_category"].keys()) | set(self.B["per_category"].keys()))
        cat_rows = []
        for c in cats:
            for t in thresholds:
                a = self._get_safe(self.A, ["per_category", c, t])
                b = self._get_safe(self.B, ["per_category", c, t])
                cat_rows.append({"category": c, "threshold": t, f"pck_{self.name_a}": a, f"pck_{self.name_b}": b, "delta": b - a})
        pd.DataFrame(cat_rows).to_csv(os.path.join(save_dir, "per_category_pck_compare.csv"), index=False)

        # per-keypoint
        kp_rows = []
        cats_kp = set(self.A["per_keypoint"].keys()) | set(self.B["per_keypoint"].keys())
        for c in sorted(cats_kp):
            kps = set(self.A["per_keypoint"].get(c, {}).keys()) | set(self.B["per_keypoint"].get(c, {}).keys())
            for kp in sorted(kps):
                for t in thresholds:
                    a = self._get_safe(self.A, ["per_keypoint", c, kp, t])
                    b = self._get_safe(self.B, ["per_keypoint", c, kp, t])
                    kp_rows.append({"category": c, "keypoint_id": kp, "threshold": t,
                                    f"pck_{self.name_a}": a, f"pck_{self.name_b}": b, "delta": b - a})
        pd.DataFrame(kp_rows).to_csv(os.path.join(save_dir, "per_keypoint_pck_compare.csv"), index=False)

        # per-image
        img_rows = []
        img_ids = sorted(set(self.A["per_image"].keys()) | set(self.B["per_image"].keys()))
        for idx in img_ids:
            for t in thresholds:
                a = self._get_safe(self.A, ["per_image", idx, t])
                b = self._get_safe(self.B, ["per_image", idx, t])
                img_rows.append({"pair_idx": idx, "threshold": t, f"pck_{self.name_a}": a, f"pck_{self.name_b}": b, "delta": b - a})
        pd.DataFrame(img_rows).to_csv(os.path.join(save_dir, "per_image_pck_compare.csv"), index=False)

        print(f"✅ Exported comparison CSVs to {save_dir}/")

    def generate_report(self, save_dir="./results_compare", threshold=0.10, top_n=15):
        """Generate a complete visual comparison report + CSVs"""
        os.makedirs(save_dir, exist_ok=True)
        print("📊 Generating comparison report...")

        self.plot_pck_curve(save_path=os.path.join(save_dir, "pck_curve_compare.png"))
        self.plot_per_category(threshold=threshold, save_path=os.path.join(save_dir, "per_category_compare.png"))
        self.plot_keypoint_difficulty(threshold=threshold, top_n=top_n,
                                      save_path=os.path.join(save_dir, "keypoint_difficulty_compare.png"))
        self.plot_image_difficulty_distribution(threshold=threshold,
                                                save_path=os.path.join(save_dir, "image_distribution_compare.png"))
        self.export_to_csv(save_dir=save_dir)

        print(f"✅ Comparison report generated in {save_dir}/")

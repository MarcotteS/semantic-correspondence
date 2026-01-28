import os
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class ResultsComparator:
    """
    Same as ResultsAnalyzer, but with two metrics (A vs B).
    Inputs can be:
      - metrics dict
      - payload dict with {"metrics": metrics}
      - path to .json (payload or metrics)
      - json string (payload or metrics)
    """

    def __init__(self, metrics_a, metrics_b, name_a="A", name_b="B"):
        self.metrics_a = self._load_metrics(metrics_a)
        self.metrics_b = self._load_metrics(metrics_b)
        self.name_a = name_a
        self.name_b = name_b

    # ---------- loading ----------
    def _load_metrics(self, x):
        if isinstance(x, dict):
            obj = x
        elif isinstance(x, str):
            s = x.strip()
            if os.path.isfile(s) and s.lower().endswith(".json"):
                with open(s, "r", encoding="utf-8") as f:
                    obj = json.load(f)
            else:
                obj = json.loads(s)
        else:
            raise TypeError(f"Expected dict / json str / .json path, got {type(x)}")

        # unwrap payload saved by evaluate_model
        if isinstance(obj, dict) and "metrics" in obj and isinstance(obj["metrics"], dict):
            obj = obj["metrics"]

        # sanity defaults
        obj.setdefault("overall", {})
        obj.setdefault("per_category", {})
        obj.setdefault("per_keypoint", {})
        obj.setdefault("per_image", {})
        return obj

    # ---------- robust threshold resolver (same logic as your analyzer) ----------
    def _resolve_threshold_key(self, d, threshold):
        keys = list(d.keys())
        if not keys:
            raise KeyError("No thresholds found.")

        if threshold in d:
            return threshold

        for fmt in (str(threshold), f"{threshold:.2f}", f"{threshold:.3f}", f"{threshold:.1f}"):
            if fmt in d:
                return fmt

        numeric = []
        for k in keys:
            try:
                numeric.append((k, float(k)))
            except Exception:
                pass

        if numeric:
            best_key = min(numeric, key=lambda kv: abs(kv[1] - float(threshold)))[0]
            return best_key

        raise KeyError(f"Could not resolve threshold {threshold}. Available keys: {keys[:10]}...")

    # ---------- PLOTS ----------
    def plot_pck_curve(self, save_path=None):
        fig, ax = plt.subplots(figsize=(10, 6))

        overall_a = self.metrics_a["overall"]
        overall_b = self.metrics_b["overall"]

        # union thresholds
        thresholds = list(set(list(overall_a.keys()) + list(overall_b.keys())))
        if len(thresholds) == 0:
            print("overall is empty in both metrics. You probably loaded the wrong JSON object.")
            return

        # sort numerically if possible
        def _as_float(x):
            try:
                return float(x)
            except Exception:
                return None

        if all(_as_float(t) is not None for t in thresholds):
            thresholds_sorted = sorted(thresholds, key=lambda x: float(x))
            x_vals = [float(t) for t in thresholds_sorted]
        else:
            thresholds_sorted = sorted(thresholds)
            x_vals = thresholds_sorted

        pck_a = [overall_a.get(t, np.nan) for t in thresholds_sorted]
        pck_b = [overall_b.get(t, np.nan) for t in thresholds_sorted]

        ax.plot(x_vals, pck_a, marker="o", linewidth=2, markersize=8, label=self.name_a)
        ax.plot(x_vals, pck_b, marker="o", linewidth=2, markersize=8, label=self.name_b)

        ax.set_xlabel("PCK Threshold (α)", fontsize=12)
        ax.set_ylabel("PCK (%)", fontsize=12)
        ax.set_title("PCK vs Threshold (Comparison)", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 100])
        ax.legend()

        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    def plot_per_category(self, threshold=0.10, save_path=None):
        fig, ax = plt.subplots(figsize=(12, 6))

        cats = sorted(set(self.metrics_a["per_category"].keys()) | set(self.metrics_b["per_category"].keys()))
        if len(cats) == 0:
            print("No categories found.")
            return

        a_vals, b_vals = [], []
        for cat in cats:
            da = self.metrics_a["per_category"].get(cat, {})
            db = self.metrics_b["per_category"].get(cat, {})

            va = np.nan
            vb = np.nan
            if da:
                ka = self._resolve_threshold_key(da, threshold)
                va = da[ka]
            if db:
                kb = self._resolve_threshold_key(db, threshold)
                vb = db[kb]

            a_vals.append(va)
            b_vals.append(vb)

        x = np.arange(len(cats))
        w = 0.38
        ax.bar(x - w/2, a_vals, width=w, alpha=0.8, label=self.name_a)
        ax.bar(x + w/2, b_vals, width=w, alpha=0.8, label=self.name_b)

        ax.set_xlabel("Category", fontsize=12)
        ax.set_ylabel(f"PCK@{float(threshold):.2f} (%)", fontsize=12)
        ax.set_title(f"Per-Category Performance (PCK@{float(threshold):.2f}) (Comparison)",
                     fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(cats, rotation=45, ha="right")
        ax.set_ylim([0, 100])
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend()

        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    def plot_keypoint_difficulty(self, threshold=0.10, top_n=15, save_path=None):
        """
        Same layout as ResultsAnalyzer (2 panels hardest/easiest),
        but shows A and B for the same keypoints (ranked by A).
        """
        kp_scores = []
        for category in self.metrics_a["per_keypoint"].keys():
            for kp_id in self.metrics_a["per_keypoint"][category].keys():
                da = self.metrics_a["per_keypoint"][category][kp_id]
                ka = self._resolve_threshold_key(da, threshold)
                a = da[ka]

                # B may be missing -> NaN
                b = np.nan
                db = self.metrics_b["per_keypoint"].get(category, {}).get(kp_id, {})
                if db:
                    kb = self._resolve_threshold_key(db, threshold)
                    b = db[kb]

                kp_label = f"{category}-{kp_id}"
                kp_scores.append((kp_label, a, b))

        if len(kp_scores) == 0:
            print("No keypoints found.")
            return

        kp_scores.sort(key=lambda x: x[1])  # by A score like analyzer
        hardest = kp_scores[:top_n]
        easiest = kp_scores[-top_n:][::-1]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        def panel(ax, items, title):
            labels = [k for k, _, _ in items]
            a_vals = [a for _, a, _ in items]
            b_vals = [b for _, _, b in items]
            y = np.arange(len(items))
            h = 0.38

            ax.barh(y - h/2, a_vals, height=h, alpha=0.8, label=self.name_a)
            ax.barh(y + h/2, b_vals, height=h, alpha=0.8, label=self.name_b)

            ax.set_yticks(y)
            ax.set_yticklabels(labels, fontsize=9)
            ax.set_xlabel(f"PCK@{float(threshold):.2f} (%)", fontsize=12)
            ax.set_title(title, fontsize=14, fontweight="bold")
            ax.set_xlim([0, 100])
            ax.grid(True, axis="x", alpha=0.3)

        panel(ax1, hardest, f"Top {top_n} Hardest Keypoints")
        panel(ax2, easiest, f"Top {top_n} Easiest Keypoints")
        ax1.legend(); ax2.legend()

        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    def plot_image_difficulty_distribution(self, threshold=0.10, save_path=None):
        scores_a = []
        for idx in self.metrics_a["per_image"].keys():
            d = self.metrics_a["per_image"][idx]
            k = self._resolve_threshold_key(d, threshold)
            scores_a.append(d[k])

        scores_b = []
        for idx in self.metrics_b["per_image"].keys():
            d = self.metrics_b["per_image"][idx]
            k = self._resolve_threshold_key(d, threshold)
            scores_b.append(d[k])

        if len(scores_a) == 0 and len(scores_b) == 0:
            print("No per-image scores found.")
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        if len(scores_a) > 0:
            ax.hist(scores_a, bins=30, alpha=0.5, edgecolor="black", label=self.name_a)
            ax.axvline(np.mean(scores_a), linestyle="--", linewidth=2,
                       label=f"{self.name_a} mean: {np.mean(scores_a):.1f}%")
        if len(scores_b) > 0:
            ax.hist(scores_b, bins=30, alpha=0.5, edgecolor="black", label=self.name_b)
            ax.axvline(np.mean(scores_b), linestyle="--", linewidth=2,
                       label=f"{self.name_b} mean: {np.mean(scores_b):.1f}%")

        ax.set_xlabel(f"PCK@{float(threshold):.2f} (%)", fontsize=12)
        ax.set_ylabel("Number of Image Pairs", fontsize=12)
        ax.set_title("Distribution of Per-Image PCK Scores (Comparison)", fontsize=14, fontweight="bold")
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    def generate_report(self, save_dir="./results_compare", threshold=0.10, top_n=15):
        os.makedirs(save_dir, exist_ok=True)
        print(" Generating comparison report...")

        self.plot_pck_curve(save_path=f"{save_dir}/pck_curve_compare.png")
        self.plot_per_category(threshold=threshold, save_path=f"{save_dir}/per_category_compare.png")
        self.plot_keypoint_difficulty(threshold=threshold, top_n=top_n, save_path=f"{save_dir}/keypoint_difficulty_compare.png")
        self.plot_image_difficulty_distribution(threshold=threshold, save_path=f"{save_dir}/image_distribution_compare.png")

        print(f" Report generated in {save_dir}/")

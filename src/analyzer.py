import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class ResultsAnalyzer:
    """
    Analyze and visualize correspondence evaluation results
    """

    def __init__(self, metrics):
        """
        Args:
            metrics: dict returned by CorrespondenceEvaluator.get_metrics()
        """
        self.metrics = metrics

    def _resolve_threshold_key(self, d, threshold):
        """Return the best matching key in dict `d` for the requested `threshold`."""
        keys = list(d.keys())
        if not keys:
            raise KeyError("No thresholds found.")

        # direct hit
        if threshold in d:
            return threshold

        # try string forms
        for fmt in (str(threshold), f"{threshold:.2f}", f"{threshold:.3f}", f"{threshold:.1f}"):
            if fmt in d:
                return fmt

        # try numeric coercion (handles '0.10' keys etc.)
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

    def plot_pck_curve(self, save_path=None):
        """Plot PCK values across different thresholds (robuste si keys sont str/float)"""
        fig, ax = plt.subplots(figsize=(10, 6))

        overall = self.metrics["overall"]
        thresholds = list(overall.keys())

        # try to sort numerically if possible
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

        pck_values = [overall[t] for t in thresholds_sorted]

        ax.plot(x_vals, pck_values, marker="o", linewidth=2, markersize=8)
        ax.set_xlabel("PCK Threshold (α)", fontsize=12)
        ax.set_ylabel("PCK (%)", fontsize=12)
        ax.set_title("PCK vs Threshold", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 100])

        for x, pck in zip(x_vals, pck_values):
            ax.text(x, pck + 2, f"{pck:.1f}%", ha="center", fontsize=9)

        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    def plot_per_category(self, threshold=0.10, save_path=None):
        """Plot PCK per category (robuste seuil str/float)"""
        fig, ax = plt.subplots(figsize=(12, 6))

        categories = sorted(self.metrics["per_category"].keys())
        pck_values = []
        for cat in categories:
            d = self.metrics["per_category"][cat]
            thr_key = self._resolve_threshold_key(d, threshold)
            pck_values.append(d[thr_key])

        colors = plt.cm.viridis(np.linspace(0, 1, len(categories)))
        bars = ax.bar(range(len(categories)), pck_values, color=colors, alpha=0.8)

        ax.set_xlabel("Category", fontsize=12)
        ax.set_ylabel(f"PCK@{float(threshold):.2f} (%)", fontsize=12)
        ax.set_title(f"Per-Category Performance (PCK@{float(threshold):.2f})",
                     fontsize=14, fontweight="bold")
        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels(categories, rotation=45, ha="right")
        ax.set_ylim([0, 100])
        ax.grid(True, axis="y", alpha=0.3)

        for bar, val in zip(bars, pck_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=9)

        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    def plot_keypoint_difficulty(self, threshold=0.10, top_n=15, save_path=None):
        """Plot keypoint difficulty (top easiest and hardest) (robuste seuil str/float)"""
        kp_scores = []
        for category in self.metrics["per_keypoint"].keys():
            for kp_id in self.metrics["per_keypoint"][category].keys():
                d = self.metrics["per_keypoint"][category][kp_id]
                thr_key = self._resolve_threshold_key(d, threshold)
                pck = d[thr_key]
                kp_label = f"{category}-{kp_id}"
                kp_scores.append((kp_label, pck))

        if len(kp_scores) == 0:
            print("No keypoints found.")
            return

        kp_scores.sort(key=lambda x: x[1])

        hardest = kp_scores[:top_n]
        easiest = kp_scores[-top_n:][::-1]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        kp_labels_hard = [kp for kp, _ in hardest]
        scores_hard = [score for _, score in hardest]
        ax1.barh(range(len(hardest)), scores_hard, alpha=0.7)
        ax1.set_yticks(range(len(hardest)))
        ax1.set_yticklabels(kp_labels_hard, fontsize=9)
        ax1.set_xlabel(f"PCK@{float(threshold):.2f} (%)", fontsize=12)
        ax1.set_title(f"Top {top_n} Hardest Keypoints", fontsize=14, fontweight="bold")
        ax1.set_xlim([0, 100])
        ax1.grid(True, axis="x", alpha=0.3)
        for i, val in enumerate(scores_hard):
            ax1.text(val + 2, i, f"{val:.1f}%", va="center", fontsize=9)

        kp_labels_easy = [kp for kp, _ in easiest]
        scores_easy = [score for _, score in easiest]
        ax2.barh(range(len(easiest)), scores_easy, alpha=0.7)
        ax2.set_yticks(range(len(easiest)))
        ax2.set_yticklabels(kp_labels_easy, fontsize=9)
        ax2.set_xlabel(f"PCK@{float(threshold):.2f} (%)", fontsize=12)
        ax2.set_title(f"Top {top_n} Easiest Keypoints", fontsize=14, fontweight="bold")
        ax2.set_xlim([0, 100])
        ax2.grid(True, axis="x", alpha=0.3)
        for i, val in enumerate(scores_easy):
            ax2.text(val + 2, i, f"{val:.1f}%", va="center", fontsize=9)

        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    def plot_image_difficulty_distribution(self, threshold=0.10, save_path=None):
        """Plot distribution of per-image PCK scores (robuste seuil str/float)"""
        image_scores = []
        for idx in self.metrics["per_image"].keys():
            d = self.metrics["per_image"][idx]
            thr_key = self._resolve_threshold_key(d, threshold)
            image_scores.append(d[thr_key])

        if len(image_scores) == 0:
            print("No per-image scores found.")
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.hist(image_scores, bins=30, alpha=0.7, edgecolor="black")
        ax.axvline(np.mean(image_scores), linestyle="--",
                   linewidth=2, label=f"Mean: {np.mean(image_scores):.1f}%")
        ax.axvline(np.median(image_scores), linestyle="--",
                   linewidth=2, label=f"Median: {np.median(image_scores):.1f}%")

        ax.set_xlabel(f"PCK@{float(threshold):.2f} (%)", fontsize=12)
        ax.set_ylabel("Number of Image Pairs", fontsize=12)
        ax.set_title("Distribution of Per-Image PCK Scores",
                     fontsize=14, fontweight="bold")
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    def plot_per_category_keypoints(self, category, threshold=0.10, save_path=None):
        """Plot keypoint performance for a specific category (robuste seuil str/float)"""
        if category not in self.metrics["per_keypoint"]:
            print(f"Category '{category}' not found in results.")
            return

        kp_data = self.metrics["per_keypoint"][category]

        # sort by threshold score (resolved)
        def _score(kp_id):
            d = kp_data[kp_id]
            thr_key = self._resolve_threshold_key(d, threshold)
            return d[thr_key]

        kp_ids = sorted(kp_data.keys(), key=_score, reverse=True)
        pck_values = [_score(kp_id) for kp_id in kp_ids]

        fig, ax = plt.subplots(figsize=(12, 6))

        colors = plt.cm.RdYlGn(np.array(pck_values) / 100)
        bars = ax.bar(range(len(kp_ids)), pck_values, color=colors, alpha=0.8, edgecolor="black")

        ax.set_xlabel("Keypoint ID", fontsize=12)
        ax.set_ylabel(f"PCK@{float(threshold):.2f} (%)", fontsize=12)
        ax.set_title(f"Per-Keypoint Performance for {category.capitalize()}",
                     fontsize=14, fontweight="bold")
        ax.set_xticks(range(len(kp_ids)))
        ax.set_xticklabels(kp_ids, rotation=0)
        ax.set_ylim([0, 100])
        ax.grid(True, axis="y", alpha=0.3)

        for bar, val in zip(bars, pck_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=9)

        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    def create_summary_table(self, threshold=0.10):
        """Create a pandas DataFrame with summary statistics (robuste seuil str/float)"""
        thr_key_overall = self._resolve_threshold_key(self.metrics["overall"], threshold)
        data = {
            "Overall PCK": [self.metrics["overall"][thr_key_overall]],
        }

        for cat in sorted(self.metrics["per_category"].keys()):
            d = self.metrics["per_category"][cat]
            thr_key = self._resolve_threshold_key(d, threshold)
            data[f"{cat}"] = [d[thr_key]]

        return pd.DataFrame(data)

    def export_to_csv(self, save_dir="./results"):
        """Export all metrics to CSV files (robuste seuil str/float)"""
        os.makedirs(save_dir, exist_ok=True)

        overall = self.metrics["overall"]
        thresholds = list(overall.keys())

        # sort thresholds numerically if possible
        def _as_float(x):
            try:
                return float(x)
            except Exception:
                return None

        if all(_as_float(t) is not None for t in thresholds):
            thresholds_sorted = sorted(thresholds, key=lambda x: float(x))
        else:
            thresholds_sorted = sorted(thresholds)

        # Overall PCK
        overall_df = pd.DataFrame([
            {"threshold": t, "pck": overall[t]}
            for t in thresholds_sorted
        ])
        overall_df.to_csv(f"{save_dir}/overall_pck.csv", index=False)

        # Per-category PCK
        category_data = []
        for cat in sorted(self.metrics["per_category"].keys()):
            cat_dict = self.metrics["per_category"][cat]
            for t in thresholds_sorted:
                # resolve using numeric value of t if possible, else try t directly
                try:
                    t_val = float(t)
                except Exception:
                    t_val = t
                thr_key = self._resolve_threshold_key(cat_dict, t_val)
                category_data.append({
                    "category": cat,
                    "threshold": t,
                    "pck": cat_dict[thr_key],
                })
        pd.DataFrame(category_data).to_csv(f"{save_dir}/per_category_pck.csv", index=False)

        # Per-keypoint PCK (with category)
        keypoint_data = []
        for category in sorted(self.metrics["per_keypoint"].keys()):
            for kp_id in sorted(self.metrics["per_keypoint"][category].keys()):
                kp_dict = self.metrics["per_keypoint"][category][kp_id]
                for t in thresholds_sorted:
                    try:
                        t_val = float(t)
                    except Exception:
                        t_val = t
                    thr_key = self._resolve_threshold_key(kp_dict, t_val)
                    keypoint_data.append({
                        "category": category,
                        "keypoint_id": kp_id,
                        "threshold": t,
                        "pck": kp_dict[thr_key],
                    })
        pd.DataFrame(keypoint_data).to_csv(f"{save_dir}/per_keypoint_pck.csv", index=False)

        # Per-image PCK
        image_data = []
        for idx in sorted(self.metrics["per_image"].keys()):
            img_dict = self.metrics["per_image"][idx]
            for t in thresholds_sorted:
                try:
                    t_val = float(t)
                except Exception:
                    t_val = t
                thr_key = self._resolve_threshold_key(img_dict, t_val)
                image_data.append({
                    "pair_idx": idx,
                    "threshold": t,
                    "pck": img_dict[thr_key],
                })
        pd.DataFrame(image_data).to_csv(f"{save_dir}/per_image_pck.csv", index=False)

        print(f"✅ Exported all metrics to {save_dir}/")

    def generate_report(self, save_dir="./results"):
        """Generate a complete visual report"""
        os.makedirs(save_dir, exist_ok=True)

        print("📊 Generating visual report...")

        self.plot_pck_curve(save_path=f"{save_dir}/pck_curve.png")
        self.plot_per_category(save_path=f"{save_dir}/per_category.png")
        self.plot_keypoint_difficulty(save_path=f"{save_dir}/keypoint_difficulty.png")
        self.plot_image_difficulty_distribution(save_path=f"{save_dir}/image_distribution.png")
        self.export_to_csv(save_dir=save_dir)

        print(f"✅ Report generated in {save_dir}/")

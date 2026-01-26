import os
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class ResultsComparator:
    """
    Same plots as ResultsAnalyzer, but comparing TWO runs (A vs B).
    Accepts:
      - metrics dict
      - payload dict that contains {"metrics": metrics}
      - JSON string of either metrics or payload
      - path to .json of either metrics or payload
    """

    def __init__(self, metrics_a, metrics_b, name_a="baseline", name_b="fine_tuned"):
        self.A = self._unwrap_metrics(self._load_any(metrics_a))
        self.B = self._unwrap_metrics(self._load_any(metrics_b))
        self.name_a = name_a
        self.name_b = name_b

    # -------- IO --------
    @staticmethod
    def _load_any(x):
        if isinstance(x, dict):
            return x
        if isinstance(x, str):
            s = x.strip()
            if os.path.isfile(s) and s.lower().endswith(".json"):
                with open(s, "r", encoding="utf-8") as f:
                    return json.load(f)
            return json.loads(s)
        raise TypeError(f"Expected dict / json str / .json path. Got {type(x)}")

    @staticmethod
    def _unwrap_metrics(obj):
        # if it's the payload saved by evaluate_model -> metrics are under "metrics"
        if isinstance(obj, dict) and "metrics" in obj and isinstance(obj["metrics"], dict):
            obj = obj["metrics"]

        # minimal sanity: ensure keys exist
        if not isinstance(obj, dict):
            raise TypeError("Loaded object is not a dict.")
        obj.setdefault("overall", {})
        obj.setdefault("per_category", {})
        obj.setdefault("per_keypoint", {})
        obj.setdefault("per_image", {})
        return obj

    # -------- threshold resolver (same spirit as your analyzer) --------
    def _resolve_threshold_key(self, d, threshold):
        keys = list((d or {}).keys())
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

    # -------- save/show --------
    def _save_show(self, fig, save_path):
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    # ================= PLOTS =================
    def plot_pck_curve(self, save_path=None):
        fig, ax = plt.subplots(figsize=(10, 6))

        oa = self.A["overall"]
        ob = self.B["overall"]

        # union keys then sort numerically if possible
        keys_union = list(set(list(oa.keys()) + list(ob.keys())))
        if len(keys_union) == 0:
            print("❌ overall is empty in both A and B. (did you pass the correct JSON?)")
            return

        def _as_float(x):
            try:
                return float(x)
            except Exception:
                return None

        if all(_as_float(k) is not None for k in keys_union):
            thr_sorted = sorted(keys_union, key=lambda x: float(x))
            x_vals = [float(t) for t in thr_sorted]
        else:
            thr_sorted = sorted(keys_union)
            x_vals = thr_sorted

        ya = [oa.get(t, np.nan) for t in thr_sorted]
        yb = [ob.get(t, np.nan) for t in thr_sorted]

        ax.plot(x_vals, ya, marker="o", linewidth=2, markersize=8, label=self.name_a)
        ax.plot(x_vals, yb, marker="o", linewidth=2, markersize=8, label=self.name_b)

        ax.set_xlabel("PCK Threshold (α)", fontsize=12)
        ax.set_ylabel("PCK (%)", fontsize=12)
        ax.set_title("PCK vs Threshold (Comparison)", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 100])
        ax.legend()

        self._save_show(fig, save_path)

    def plot_per_category(self, threshold=0.10, save_path=None):
        fig, ax = plt.subplots(figsize=(12, 6))

        cats = sorted(set(self.A["per_category"].keys()) | set(self.B["per_category"].keys()))
        if len(cats) == 0:
            print("❌ per_category empty in both A and B.")
            return

        a_vals, b_vals = [], []
        for cat in cats:
            da = self.A["per_category"].get(cat, {})
            db = self.B["per_category"].get(cat, {})

            va = np.nan
            vb = np.nan
            if da:
                try:
                    ka = self._resolve_threshold_key(da, threshold)
                    va = da[ka]
                except KeyError:
                    pass
            if db:
                try:
                    kb = self._resolve_threshold_key(db, threshold)
                    vb = db[kb]
                except KeyError:
                    pass

            a_vals.append(va)
            b_vals.append(vb)

        x = np.arange(len(cats))
        w = 0.38
        bars_a = ax.bar(x - w/2, a_vals, width=w, alpha=0.8, label=self.name_a)
        bars_b = ax.bar(x + w/2, b_vals, width=w, alpha=0.8, label=self.name_b)

        ax.set_xlabel("Category", fontsize=12)
        ax.set_ylabel(f"PCK@{float(threshold):.2f} (%)", fontsize=12)
        ax.set_title(f"Per-Category Performance (PCK@{float(threshold):.2f}) (Comparison)",
                     fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(cats, rotation=45, ha="right")
        ax.set_ylim([0, 100])
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend()

        self._save_show(fig, save_path)

    def plot_keypoint_difficulty(self, threshold=0.10, top_n=15, save_path=None):
        """
        Same 2-panel layout as ResultsAnalyzer, but shows A and B for the SAME keypoints.
        We rank keypoints by A (like your analyzer does on one metrics).
        This avoids the "no overlap" problem when keys/thresholds mismatch.
        """
        kp_scores = []
        for category in self.A["per_keypoint"].keys():
            for kp_id in self.A["per_keypoint"][category].keys():
                da = self.A["per_keypoint"][category][kp_id]
                try:
                    ka = self._resolve_threshold_key(da, threshold)
                    a = da[ka]
                except KeyError:
                    continue

                # B value if exists
                b = np.nan
                db = self.B["per_keypoint"].get(category, {}).get(kp_id, {})
                if db:
                    try:
                        kb = self._resolve_threshold_key(db, threshold)
                        b = db[kb]
                    except KeyError:
                        b = np.nan

                kp_scores.append((f"{category}-{kp_id}", a, b))

        if len(kp_scores) == 0:
            print("No keypoints found (in A) for this threshold. Try another threshold.")
            return

        kp_scores.sort(key=lambda x: x[1])  # by A asc
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

        panel(ax1, hardest, f"Top {top_n} Hardest Keypoints (ranked by {self.name_a})")
        panel(ax2, easiest, f"Top {top_n} Easiest Keypoints (ranked by {self.name_a})")
        ax1.legend(); ax2.legend()

        self._save_show(fig, save_path)

    def plot_image_difficulty_distribution(self, threshold=0.10, save_path=None):
        a_scores = []
        for idx, d in self.A["per_image"].items():
            try:
                k = self._resolve_threshold_key(d, threshold)
                a_scores.append(d[k])
            except KeyError:
                pass

        b_scores = []
        for idx, d in self.B["per_image"].items():
            try:
                k = self._resolve_threshold_key(d, threshold)
                b_scores.append(d[k])
            except KeyError:
                pass

        if len(a_scores) == 0 and len(b_scores) == 0:
            print("No per-image scores found.")
            return

        fig, ax = plt.subplots(figsize=(10, 6))
        if len(a_scores) > 0:
            ax.hist(a_scores, bins=30, alpha=0.5, edgecolor="black", label=self.name_a)
            ax.axvline(np.mean(a_scores), linestyle="--", linewidth=2,
                       label=f"{self.name_a} mean: {np.mean(a_scores):.1f}%")
        if len(b_scores) > 0:
            ax.hist(b_scores, bins=30, alpha=0.5, edgecolor="black", label=self.name_b)
            ax.axvline(np.mean(b_scores), linestyle="--", linewidth=2,
                       label=f"{self.name_b} mean: {np.mean(b_scores):.1f}%")

        ax.set_xlabel(f"PCK@{float(threshold):.2f} (%)", fontsize=12)
        ax.set_ylabel("Number of Image Pairs", fontsize=12)
        ax.set_title("Distribution of Per-Image PCK Scores (Comparison)", fontsize=14, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        self._save_show(fig, save_path)

    def generate_report(self, save_dir="./results_compare", threshold=0.10, top_n=15):
        os.makedirs(save_dir, exist_ok=True)
        print("📊 Generating comparison report...")

        self.plot_pck_curve(save_path=f"{save_dir}/pck_curve_compare.png")
        self.plot_per_category(threshold=threshold, save_path=f"{save_dir}/per_category_compare.png")
        self.plot_keypoint_difficulty(threshold=threshold, top_n=top_n,
                                      save_path=f"{save_dir}/keypoint_difficulty_compare.png")
        self.plot_image_difficulty_distribution(threshold=threshold,
                                                save_path=f"{save_dir}/image_distribution_compare.png")

        print(f"✅ Report generated in {save_dir}/")

import os
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class ResultsComparator:
    """
    Same plots as ResultsAnalyzer, but comparing TWO metrics (A vs B).
    Inputs can be:
      - dict (metrics)
      - JSON string
      - path to .json
    """

    def __init__(self, metrics_a, metrics_b, name_a="baseline", name_b="fine_tuned"):
        self.A = self._load_metrics(metrics_a)
        self.B = self._load_metrics(metrics_b)
        self.name_a = name_a
        self.name_b = name_b

    # ---------------- IO ----------------
    @staticmethod
    def _load_metrics(x):
        if isinstance(x, dict):
            return x
        if isinstance(x, str):
            s = x.strip()
            if os.path.isfile(s) and s.lower().endswith(".json"):
                with open(s, "r", encoding="utf-8") as f:
                    return json.load(f)
            return json.loads(s)
        raise TypeError(f"metrics must be dict / json string / .json path. Got {type(x)}")

    # ------------- robust threshold resolver (same idea as your analyzer) -------------
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

    # ---------------- helpers ----------------
    def _save_show(self, fig, save_path):
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    @staticmethod
    def _sorted_thresholds(overall_dict):
        thresholds = list(overall_dict.keys())

        def _as_float(x):
            try:
                return float(x)
            except Exception:
                return None

        if thresholds and all(_as_float(t) is not None for t in thresholds):
            thresholds_sorted = sorted(thresholds, key=lambda x: float(x))
            x_vals = [float(t) for t in thresholds_sorted]
        else:
            thresholds_sorted = sorted(thresholds)
            x_vals = thresholds_sorted
        return thresholds_sorted, x_vals

    # ---------------- plots (same as ResultsAnalyzer, but with 2 lines/bars) ----------------
    def plot_pck_curve(self, save_path=None):
        fig, ax = plt.subplots(figsize=(10, 6))

        overall_a = self.A["overall"]
        overall_b = self.B["overall"]

        # union of thresholds (robust)
        keys_union = list(set(list(overall_a.keys()) + list(overall_b.keys())))

        def _as_float(x):
            try:
                return float(x)
            except Exception:
                return None

        if keys_union and all(_as_float(t) is not None for t in keys_union):
            thresholds_sorted = sorted(keys_union, key=lambda x: float(x))
            x_vals = [float(t) for t in thresholds_sorted]
        else:
            thresholds_sorted = sorted(keys_union)
            x_vals = thresholds_sorted

        y_a = [overall_a.get(t, np.nan) for t in thresholds_sorted]
        y_b = [overall_b.get(t, np.nan) for t in thresholds_sorted]

        ax.plot(x_vals, y_a, marker="o", linewidth=2, markersize=8, label=self.name_a)
        ax.plot(x_vals, y_b, marker="o", linewidth=2, markersize=8, label=self.name_b)

        ax.set_xlabel("PCK Threshold (α)", fontsize=12)
        ax.set_ylabel("PCK (%)", fontsize=12)
        ax.set_title("PCK vs Threshold (Comparison)", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 100])
        ax.legend()

        # annotate both values + delta (optional, compact)
        for x, a, b in zip(x_vals, y_a, y_b):
            if np.isfinite(a):
                ax.text(x, a + 2, f"{a:.1f}", ha="center", fontsize=8)
            if np.isfinite(b):
                ax.text(x, b - 6, f"{b:.1f}", ha="center", fontsize=8)
            if np.isfinite(a) and np.isfinite(b):
                ax.text(x, max(a, b) + 6, f"Δ{(b-a):+.1f}", ha="center", fontsize=8)

        self._save_show(fig, save_path)

    def plot_per_category(self, threshold=0.10, save_path=None):
        fig, ax = plt.subplots(figsize=(12, 6))

        cats = sorted(set(self.A["per_category"].keys()) | set(self.B["per_category"].keys()))

        a_vals, b_vals = [], []
        for cat in cats:
            da = self.A["per_category"].get(cat, {})
            db = self.B["per_category"].get(cat, {})

            va = np.nan
            vb = np.nan
            if da:
                try:
                    ka = self._resolve_threshold_key(da, threshold)
                    va = da.get(ka, np.nan)
                except KeyError:
                    va = np.nan
            if db:
                try:
                    kb = self._resolve_threshold_key(db, threshold)
                    vb = db.get(kb, np.nan)
                except KeyError:
                    vb = np.nan

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

        # annotate values + delta
        for i, (ba, bb) in enumerate(zip(bars_a, bars_b)):
            va = ba.get_height()
            vb = bb.get_height()
            if np.isfinite(va):
                ax.text(ba.get_x() + ba.get_width()/2, va + 1, f"{va:.1f}", ha="center", fontsize=8)
            if np.isfinite(vb):
                ax.text(bb.get_x() + bb.get_width()/2, vb + 1, f"{vb:.1f}", ha="center", fontsize=8)
            if np.isfinite(va) and np.isfinite(vb):
                ax.text(i, max(va, vb) + 6, f"Δ{(vb-va):+.1f}", ha="center", fontsize=8)

        self._save_show(fig, save_path)

    def plot_keypoint_difficulty(self, threshold=0.10, top_n=15, save_path=None):
        """
        Same layout as analyzer (two panels hardest/easiest),
        but ranks by (A+B)/2 when both exist, and shows A/B/Δ labels.
        """
        kp_items = []  # (label, a, b, avg)
        cats = set(self.A["per_keypoint"].keys()) | set(self.B["per_keypoint"].keys())
        for cat in cats:
            kA = self.A["per_keypoint"].get(cat, {})
            kB = self.B["per_keypoint"].get(cat, {})
            kp_ids = set(kA.keys()) | set(kB.keys())
            for kp_id in kp_ids:
                da = kA.get(kp_id, {})
                db = kB.get(kp_id, {})
                va = vb = np.nan

                if da:
                    try:
                        ka = self._resolve_threshold_key(da, threshold)
                        va = da.get(ka, np.nan)
                    except KeyError:
                        va = np.nan
                if db:
                    try:
                        kb = self._resolve_threshold_key(db, threshold)
                        vb = db.get(kb, np.nan)
                    except KeyError:
                        vb = np.nan

                if np.isfinite(va) or np.isfinite(vb):
                    avg = np.nanmean([va, vb])
                    kp_items.append((f"{cat}-{kp_id}", va, vb, avg))

        if len(kp_items) == 0:
            print("No keypoints found.")
            return

        kp_items.sort(key=lambda x: x[3])  # by avg asc
        hardest = kp_items[:top_n]
        easiest = kp_items[-top_n:][::-1]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        def _panel(ax, items, title):
            labels = [x[0] for x in items]
            av = [x[1] for x in items]
            bv = [x[2] for x in items]
            y = np.arange(len(items))
            h = 0.38

            ax.barh(y - h/2, av, height=h, alpha=0.8, label=self.name_a)
            ax.barh(y + h/2, bv, height=h, alpha=0.8, label=self.name_b)

            ax.set_yticks(y)
            ax.set_yticklabels(labels, fontsize=9)
            ax.set_xlabel(f"PCK@{float(threshold):.2f} (%)", fontsize=12)
            ax.set_title(title, fontsize=14, fontweight="bold")
            ax.set_xlim([0, 100])
            ax.grid(True, axis="x", alpha=0.3)

            for i, (a, b) in enumerate(zip(av, bv)):
                txt = []
                if np.isfinite(a): txt.append(f"A:{a:.1f}")
                if np.isfinite(b): txt.append(f"B:{b:.1f}")
                if np.isfinite(a) and np.isfinite(b): txt.append(f"Δ{(b-a):+.1f}")
                if txt:
                    ax.text((np.nanmax([a, b]) if (np.isfinite(a) or np.isfinite(b)) else 0) + 2,
                            i, "  ".join(txt), va="center", fontsize=8)

        _panel(ax1, hardest, f"Top {top_n} Hardest Keypoints (Comparison)")
        _panel(ax2, easiest, f"Top {top_n} Easiest Keypoints (Comparison)")

        ax1.legend()
        ax2.legend()
        self._save_show(fig, save_path)

    def plot_image_difficulty_distribution(self, threshold=0.10, save_path=None):
        """
        Same idea as analyzer histogram, but overlays A and B.
        """
        scores_a = []
        for idx, d in (self.A.get("per_image", {}) or {}).items():
            try:
                k = self._resolve_threshold_key(d, threshold)
                scores_a.append(d[k])
            except Exception:
                pass

        scores_b = []
        for idx, d in (self.B.get("per_image", {}) or {}).items():
            try:
                k = self._resolve_threshold_key(d, threshold)
                scores_b.append(d[k])
            except Exception:
                pass

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
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        self._save_show(fig, save_path)

    def plot_per_category_keypoints(self, category, threshold=0.10, save_path=None):
        """
        Same as analyzer (bar plot of keypoints), but with two bars per keypoint.
        Uses union of keypoints in that category.
        """
        if category not in self.A["per_keypoint"] and category not in self.B["per_keypoint"]:
            print(f"Category '{category}' not found in results.")
            return

        kpA = self.A["per_keypoint"].get(category, {})
        kpB = self.B["per_keypoint"].get(category, {})
        kp_ids = sorted(set(kpA.keys()) | set(kpB.keys()))

        a_vals, b_vals = [], []
        for kp_id in kp_ids:
            da = kpA.get(kp_id, {})
            db = kpB.get(kp_id, {})
            va = vb = np.nan

            if da:
                try:
                    ka = self._resolve_threshold_key(da, threshold)
                    va = da.get(ka, np.nan)
                except KeyError:
                    pass
            if db:
                try:
                    kb = self._resolve_threshold_key(db, threshold)
                    vb = db.get(kb, np.nan)
                except KeyError:
                    pass

            a_vals.append(va)
            b_vals.append(vb)

        x = np.arange(len(kp_ids))
        w = 0.38
        fig, ax = plt.subplots(figsize=(12, 6))
        bars_a = ax.bar(x - w/2, a_vals, width=w, alpha=0.8, label=self.name_a)
        bars_b = ax.bar(x + w/2, b_vals, width=w, alpha=0.8, label=self.name_b)

        ax.set_xlabel("Keypoint ID", fontsize=12)
        ax.set_ylabel(f"PCK@{float(threshold):.2f} (%)", fontsize=12)
        ax.set_title(f"Per-Keypoint Performance for {category.capitalize()} (Comparison)",
                     fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(kp_ids, rotation=0)
        ax.set_ylim([0, 100])
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend()

        for i, (ba, bb) in enumerate(zip(bars_a, bars_b)):
            va = ba.get_height()
            vb = bb.get_height()
            if np.isfinite(va):
                ax.text(ba.get_x() + ba.get_width()/2., va + 1, f"{va:.1f}", ha="center", fontsize=8)
            if np.isfinite(vb):
                ax.text(bb.get_x() + bb.get_width()/2., vb + 1, f"{vb:.1f}", ha="center", fontsize=8)
            if np.isfinite(va) and np.isfinite(vb):
                ax.text(i, max(va, vb) + 6, f"Δ{(vb-va):+.1f}", ha="center", fontsize=8)

        self._save_show(fig, save_path)

    # ---------------- table + export ----------------
    def create_summary_table(self, threshold=0.10):
        """Like analyzer but with A, B, Δ columns."""
        thrA = self._resolve_threshold_key(self.A["overall"], threshold)
        thrB = self._resolve_threshold_key(self.B["overall"], threshold)

        data = {
            "Overall A": [self.A["overall"][thrA]],
            "Overall B": [self.B["overall"][thrB]],
            "Δ Overall": [self.B["overall"][thrB] - self.A["overall"][thrA]],
        }

        cats = sorted(set(self.A["per_category"].keys()) | set(self.B["per_category"].keys()))
        for cat in cats:
            da = self.A["per_category"].get(cat, {})
            db = self.B["per_category"].get(cat, {})
            va = vb = np.nan
            if da:
                try:
                    ka = self._resolve_threshold_key(da, threshold)
                    va = da.get(ka, np.nan)
                except KeyError:
                    pass
            if db:
                try:
                    kb = self._resolve_threshold_key(db, threshold)
                    vb = db.get(kb, np.nan)
                except KeyError:
                    pass

            data[f"{cat} A"] = [va]
            data[f"{cat} B"] = [vb]
            data[f"{cat} Δ"] = [vb - va if np.isfinite(va) and np.isfinite(vb) else np.nan]

        return pd.DataFrame(data)

    def export_to_csv(self, save_dir="./results_compare"):
        """Export comparison CSVs (overall, per-category, per-keypoint, per-image)"""
        os.makedirs(save_dir, exist_ok=True)

        # --- overall
        oa = self.A["overall"]
        ob = self.B["overall"]
        thr_union = sorted(set(list(oa.keys()) + list(ob.keys())), key=lambda x: float(x) if str(x).replace('.','',1).isdigit() else str(x))

        rows = []
        for t in thr_union:
            a = oa.get(t, np.nan)
            b = ob.get(t, np.nan)
            rows.append({"threshold": t, f"pck_{self.name_a}": a, f"pck_{self.name_b}": b, "delta": b - a})
        pd.DataFrame(rows).to_csv(os.path.join(save_dir, "overall_pck_compare.csv"), index=False)

        print(f"✅ Exported comparison CSVs to {save_dir}/")

    def generate_report(self, save_dir="./results_compare", threshold=0.10, top_n=15):
        """Generate a complete visual report (same set as analyzer, but comparative)"""
        os.makedirs(save_dir, exist_ok=True)

        print("📊 Generating comparison report...")

        self.plot_pck_curve(save_path=f"{save_dir}/pck_curve_compare.png")
        self.plot_per_category(threshold=threshold, save_path=f"{save_dir}/per_category_compare.png")
        self.plot_keypoint_difficulty(threshold=threshold, top_n=top_n, save_path=f"{save_dir}/keypoint_difficulty_compare.png")
        self.plot_image_difficulty_distribution(threshold=threshold, save_path=f"{save_dir}/image_distribution_compare.png")
        self.export_to_csv(save_dir=save_dir)

        print(f"✅ Report generated in {save_dir}/")

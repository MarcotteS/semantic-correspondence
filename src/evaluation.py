import torch
import numpy as np
from collections import defaultdict
from tqdm import tqdm


class CorrespondenceEvaluator:
    """
    Evaluates semantic correspondence using PCK metric
    """

    def __init__(self, thresholds=[0.05, 0.10, 0.15, 0.20]):
        """
        Args:
            thresholds: List of PCK thresholds (relative to bbox size)
        """
        self.thresholds = thresholds
        self.reset()

    def reset(self):
        """Reset all metrics"""
        # Overall metrics: {threshold: [correct_count, total_count]}
        self.overall = {t: [0, 0] for t in self.thresholds}

        # Per-keypoint metrics: {(category, kp_id): {threshold: [correct, total]}}
        self.per_keypoint = defaultdict(lambda: {t: [0, 0] for t in self.thresholds})

        # Per-image metrics: {pair_idx: {threshold: [correct, total]}}
        self.per_image = defaultdict(lambda: {t: [0, 0] for t in self.thresholds})

        # Per-category metrics: {category: {threshold: [correct, total]}}
        self.per_category = defaultdict(lambda: {t: [0, 0] for t in self.thresholds})

    def compute_pck(self, pred_kps, gt_kps, threshold_value, n_pts):
        """
        Compute PCK for a single sample

        Args:
            pred_kps: torch.Tensor [N, 2] - predicted keypoints (x, y)
            gt_kps: torch.Tensor [N, 2] - ground truth keypoints (x, y)
            threshold_value: float - absolute distance threshold (in pixels)
            n_pts: int - number of valid keypoints

        Returns:
            correct: torch.Tensor [N] - binary mask of correct predictions
            distances: torch.Tensor [N] - L2 distances for valid keypoints
        """
        # Compute L2 distance
        distances = torch.norm(pred_kps - gt_kps, dim=1)  # [N]

        # Check if within threshold
        correct = (distances < threshold_value).float()  # [N]

        # Only consider valid keypoints
        correct[n_pts:] = 0
        distances[n_pts:] = float('inf')

        return correct, distances

    def update(self, pred_kps, batch):
        """
        Update metrics with predictions from a batch

        Args:
            pred_kps: torch.Tensor [N, 2] - predicted keypoints
            batch: dict containing:
                - trg_kps: ground truth target keypoints [N, 2]
                - pckthres: PCK threshold value (bbox max dimension)
                - n_pts: number of valid keypoints
                - kps_ids: list of keypoint semantic IDs (length N)
                - pair_idx: index of this image pair
                - category: category name (e.g., 'cat', 'dog')
        """
        gt_kps = batch['trg_kps']
        n_pts = batch['n_pts'].item()
        pck_thres = batch['pckthres'].item()
        kps_ids = batch['kps_ids']
        pair_idx = batch['pair_idx']
        category = batch['category']

        # Handle kps_ids being a tuple from DataLoader collation
        if isinstance(kps_ids, tuple):
            kps_ids = list(kps_ids)

        # Move to same device
        pred_kps = pred_kps.to(gt_kps.device)

        # Evaluate at each threshold
        for alpha in self.thresholds:
            threshold_value = alpha * pck_thres
            correct, distances = self.compute_pck(pred_kps, gt_kps, threshold_value, n_pts)

            n_correct = correct[:n_pts].sum().item()

            # Update overall metrics
            self.overall[alpha][0] += n_correct
            self.overall[alpha][1] += n_pts

            # Update per-image metrics
            self.per_image[pair_idx][alpha][0] += n_correct
            self.per_image[pair_idx][alpha][1] += n_pts

            # Update per-category metrics
            self.per_category[category][alpha][0] += n_correct
            self.per_category[category][alpha][1] += n_pts

            # Update per-keypoint metrics (with category prefix)
            for kp_idx in range(n_pts):
                kp_id = kps_ids[kp_idx]
                # Handle if kp_id is a tuple (from collate_fn)
                if isinstance(kp_id, tuple):
                    kp_id = kp_id[0]

                if kp_id != "-1":  # Skip padding
                    # Create category-specific keypoint identifier
                    kp_key = (category, kp_id)
                    is_correct = correct[kp_idx].item()
                    self.per_keypoint[kp_key][alpha][0] += is_correct
                    self.per_keypoint[kp_key][alpha][1] += 1

    def get_metrics(self):
        """
        Compute final metrics

        Returns:
            dict with keys:
                - 'overall': {threshold: pck_value}
                - 'per_keypoint': {kp_id: {threshold: pck_value}}
                - 'per_image': {pair_idx: {threshold: pck_value}}
                - 'per_category': {category: {threshold: pck_value}}
        """
        metrics = {}

        # Overall PCK
        metrics['overall'] = {}
        for alpha in self.thresholds:
            correct, total = self.overall[alpha]
            metrics['overall'][alpha] = 100.0 * correct / total if total > 0 else 0.0

        # Per-keypoint PCK
        metrics['per_keypoint'] = {}
        for (category, kp_id), results in self.per_keypoint.items():
            # Store as nested dict: metrics['per_keypoint'][category][kp_id]
            if category not in metrics['per_keypoint']:
                metrics['per_keypoint'][category] = {}
            metrics['per_keypoint'][category][kp_id] = {}
            for alpha in self.thresholds:
                correct, total = results[alpha]
                metrics['per_keypoint'][category][kp_id][alpha] = 100.0 * correct / total if total > 0 else 0.0

        # Per-image PCK
        metrics['per_image'] = {}
        for pair_idx, results in self.per_image.items():
            metrics['per_image'][pair_idx] = {}
            for alpha in self.thresholds:
                correct, total = results[alpha]
                metrics['per_image'][pair_idx][alpha] = 100.0 * correct / total if total > 0 else 0.0

        # Per-category PCK
        metrics['per_category'] = {}
        for category, results in self.per_category.items():
            metrics['per_category'][category] = {}
            for alpha in self.thresholds:
                correct, total = results[alpha]
                metrics['per_category'][category][alpha] = 100.0 * correct / total if total > 0 else 0.0

        return metrics

    def print_summary(self, metrics=None):
        """Print a summary of the evaluation results"""
        if metrics is None:
            metrics = self.get_metrics()

        print("\n" + "="*70)
        print("EVALUATION SUMMARY")
        print("="*70)

        # Overall results
        print("\n📊 Overall PCK:")
        print("-" * 70)
        for alpha in self.thresholds:
            pck = metrics['overall'][alpha]
            print(f"  PCK@{alpha:.2f}: {pck:.2f}%")

        # Per-category results
        print("\n📁 Per-Category PCK:")
        print("-" * 70)
        for category in sorted(metrics['per_category'].keys()):
            pck_str = " | ".join([f"{alpha:.2f}: {metrics['per_category'][category][alpha]:.2f}%"
                                   for alpha in self.thresholds])
            print(f"  {category:15s} → {pck_str}")

        # Per-keypoint results (top 10 best and worst)
        print("\n🎯 Per-Keypoint PCK (at α=0.10):")
        print("-" * 70)

        # Flatten keypoint scores with category context
        kp_scores = []
        for category in metrics['per_keypoint'].keys():
            for kp_id in metrics['per_keypoint'][category].keys():
                pck = metrics['per_keypoint'][category][kp_id][0.10]
                kp_scores.append((category, kp_id, pck))

        kp_scores.sort(key=lambda x: x[2], reverse=True)

        if len(kp_scores) > 0:
            print("  Top 10 easiest keypoints:")
            for category, kp_id, score in kp_scores[:10]:
                print(f"    {category:15s} - Keypoint {kp_id:3s}: {score:.2f}%")

            if len(kp_scores) > 10:
                print("\n  Top 10 hardest keypoints:")
                for category, kp_id, score in kp_scores[-10:]:
                    print(f"    {category:15s} - Keypoint {kp_id:3s}: {score:.2f}%")

        # Per-image statistics
        print("\n🖼️  Per-Image PCK Statistics (at α=0.10):")
        print("-" * 70)
        image_scores = [metrics['per_image'][idx][0.10] for idx in metrics['per_image'].keys()]
        if len(image_scores) > 0:
            print(f"  Mean:   {np.mean(image_scores):.2f}%")
            print(f"  Median: {np.median(image_scores):.2f}%")
            print(f"  Std:    {np.std(image_scores):.2f}%")
            print(f"  Min:    {np.min(image_scores):.2f}%")
            print(f"  Max:    {np.max(image_scores):.2f}%")

        print("\n" + "="*70)


import os
import json
import torch
from tqdm import tqdm

def evaluate_model(
    matcher,
    dataloader,
    thresholds=(0.05, 0.10, 0.15, 0.20),
    run_name=None,
    force_recalculation=False,
):
    """
    Evaluate correspondence matcher on a dataset, with Drive caching.

    - Sauvegarde les metrics en JSON sur Google Drive
    - Si le fichier existe déjà, on recharge (pas de recalcul)
    - Recalcule seulement si force_recalculation=True
    """

    from datetime import datetime
    import os, json, torch
    from tqdm import tqdm

    # ----------------- dossier metrics sur Drive -----------------
    exp_name = getattr(matcher, "exp_name", "stage2")
    run_name = run_name or exp_name

    save_dir = f"/content/drive/MyDrive/semantic-correspondance-project/metrics/{run_name}"
    os.makedirs(save_dir, exist_ok=True)

    th_str = "-".join([f"{t:.2f}" for t in thresholds])
    metrics_path = os.path.join(save_dir, f"metrics_th_{th_str}.json")

    # ----------------- helpers -----------------
    def _to_jsonable(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().tolist()
        if isinstance(x, (float, int, str, bool)) or x is None:
            return x
        if isinstance(x, dict):
            return {str(k): _to_jsonable(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [_to_jsonable(v) for v in x]
        try:
            import numpy as np
            if isinstance(x, (np.integer, np.floating)):
                return x.item()
            if isinstance(x, np.ndarray):
                return x.tolist()
        except Exception:
            pass
        return str(x)

    # ----------------- cache load -----------------
    if (not force_recalculation) and os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            payload = json.load(f)
        metrics = payload["metrics"]
        print(f"✅ Metrics chargées depuis Drive : {metrics_path}")
        return metrics

    # ----------------- compute -----------------
    evaluator = CorrespondenceEvaluator(thresholds=list(thresholds))
    matcher.extractor.model.eval()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            src_img = batch["src_img"]
            trg_img = batch["trg_img"]
            src_kps = batch["src_kps"]

            # prédictions
            pred_kps = matcher.find_correspondences(src_img, trg_img, src_kps)

            batch_size = src_img.shape[0]
            for b in range(batch_size):
                pred_kps_b = pred_kps[b]
                trg_kps_b = batch["trg_kps"][b]
                n_pts_b = batch["n_pts"][b]
                pckthres_b = batch["pckthres"][b]
                kps_ids_b = batch["kps_ids"][b]
                pair_idx_b = batch["pair_idx"][b]
                category_b = batch["category"][b]

                batch_single = {
                    "trg_kps": trg_kps_b,
                    "pckthres": pckthres_b,
                    "n_pts": n_pts_b,
                    "kps_ids": kps_ids_b,
                    "pair_idx": pair_idx_b,
                    "category": category_b,
                }

                evaluator.update(pred_kps_b, batch_single)

    metrics = evaluator.get_metrics()
    evaluator.print_summary(metrics)

    # ----------------- save on Drive -----------------
    payload = {
        "run_name": run_name,
        "exp_name": exp_name,
        "thresholds": list(thresholds),
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "metrics": _to_jsonable(metrics),
    }

    with open(metrics_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"✅ Metrics sauvegardées sur Drive : {metrics_path}")

    return metrics




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

    def plot_pck_curve(self, save_path=None):
        """Plot PCK values across different thresholds"""
        fig, ax = plt.subplots(figsize=(10, 6))

        thresholds = sorted(self.metrics['overall'].keys())
        pck_values = [self.metrics['overall'][t] for t in thresholds]

        ax.plot(thresholds, pck_values, marker='o', linewidth=2, markersize=8)
        ax.set_xlabel('PCK Threshold (α)', fontsize=12)
        ax.set_ylabel('PCK (%)', fontsize=12)
        ax.set_title('PCK vs Threshold', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 100])

        # Add value labels
        for t, pck in zip(thresholds, pck_values):
            ax.text(t, pck + 2, f'{pck:.1f}%', ha='center', fontsize=9)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_per_category(self, threshold=0.10, save_path=None):
        """Plot PCK per category"""
        fig, ax = plt.subplots(figsize=(12, 6))

        categories = sorted(self.metrics['per_category'].keys())
        pck_values = [self.metrics['per_category'][cat][threshold] for cat in categories]

        colors = plt.cm.viridis(np.linspace(0, 1, len(categories)))
        bars = ax.bar(range(len(categories)), pck_values, color=colors, alpha=0.8)

        ax.set_xlabel('Category', fontsize=12)
        ax.set_ylabel(f'PCK@{threshold:.2f} (%)', fontsize=12)
        ax.set_title(f'Per-Category Performance (PCK@{threshold:.2f})',
                     fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels(categories, rotation=45, ha='right')
        ax.set_ylim([0, 100])
        ax.grid(True, axis='y', alpha=0.3)

        # Add value labels on bars
        for bar, val in zip(bars, pck_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{val:.1f}%', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_keypoint_difficulty(self, threshold=0.10, top_n=15, save_path=None):
        """Plot keypoint difficulty (top easiest and hardest)"""
        # Flatten keypoint scores with category context
        kp_scores = []
        for category in self.metrics['per_keypoint'].keys():
            for kp_id in self.metrics['per_keypoint'][category].keys():
                pck = self.metrics['per_keypoint'][category][kp_id][threshold]
                kp_label = f"{category}-{kp_id}"
                kp_scores.append((kp_label, pck))

        kp_scores.sort(key=lambda x: x[1])

        # Get top N hardest and easiest
        hardest = kp_scores[:top_n]
        easiest = kp_scores[-top_n:][::-1]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Hardest keypoints
        kp_labels_hard = [kp for kp, _ in hardest]
        scores_hard = [score for _, score in hardest]
        ax1.barh(range(len(hardest)), scores_hard, color='crimson', alpha=0.7)
        ax1.set_yticks(range(len(hardest)))
        ax1.set_yticklabels(kp_labels_hard, fontsize=9)
        ax1.set_xlabel(f'PCK@{threshold:.2f} (%)', fontsize=12)
        ax1.set_title(f'Top {top_n} Hardest Keypoints', fontsize=14, fontweight='bold')
        ax1.set_xlim([0, 100])
        ax1.grid(True, axis='x', alpha=0.3)

        # Add value labels
        for i, val in enumerate(scores_hard):
            ax1.text(val + 2, i, f'{val:.1f}%', va='center', fontsize=9)

        # Easiest keypoints
        kp_labels_easy = [kp for kp, _ in easiest]
        scores_easy = [score for _, score in easiest]
        ax2.barh(range(len(easiest)), scores_easy, color='mediumseagreen', alpha=0.7)
        ax2.set_yticks(range(len(easiest)))
        ax2.set_yticklabels(kp_labels_easy, fontsize=9)
        ax2.set_xlabel(f'PCK@{threshold:.2f} (%)', fontsize=12)
        ax2.set_title(f'Top {top_n} Easiest Keypoints', fontsize=14, fontweight='bold')
        ax2.set_xlim([0, 100])
        ax2.grid(True, axis='x', alpha=0.3)

        # Add value labels
        for i, val in enumerate(scores_easy):
            ax2.text(val + 2, i, f'{val:.1f}%', va='center', fontsize=9)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_image_difficulty_distribution(self, threshold=0.10, save_path=None):
        """Plot distribution of per-image PCK scores"""
        image_scores = [self.metrics['per_image'][idx][threshold]
                       for idx in self.metrics['per_image'].keys()]

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.hist(image_scores, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(image_scores), color='red', linestyle='--',
                  linewidth=2, label=f'Mean: {np.mean(image_scores):.1f}%')
        ax.axvline(np.median(image_scores), color='green', linestyle='--',
                  linewidth=2, label=f'Median: {np.median(image_scores):.1f}%')

        ax.set_xlabel(f'PCK@{threshold:.2f} (%)', fontsize=12)
        ax.set_ylabel('Number of Image Pairs', fontsize=12)
        ax.set_title('Distribution of Per-Image PCK Scores',
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_per_category_keypoints(self, category, threshold=0.10, save_path=None):
        """Plot keypoint performance for a specific category"""
        if category not in self.metrics['per_keypoint']:
            print(f"Category '{category}' not found in results.")
            return

        kp_data = self.metrics['per_keypoint'][category]
        kp_ids = sorted(kp_data.keys(), key=lambda x: kp_data[x][threshold], reverse=True)
        pck_values = [kp_data[kp_id][threshold] for kp_id in kp_ids]

        fig, ax = plt.subplots(figsize=(12, 6))

        colors = plt.cm.RdYlGn(np.array(pck_values) / 100)
        bars = ax.bar(range(len(kp_ids)), pck_values, color=colors, alpha=0.8, edgecolor='black')

        ax.set_xlabel('Keypoint ID', fontsize=12)
        ax.set_ylabel(f'PCK@{threshold:.2f} (%)', fontsize=12)
        ax.set_title(f'Per-Keypoint Performance for {category.capitalize()}',
                     fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(kp_ids)))
        ax.set_xticklabels(kp_ids, rotation=0)
        ax.set_ylim([0, 100])
        ax.grid(True, axis='y', alpha=0.3)

        # Add value labels on bars
        for bar, val in zip(bars, pck_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{val:.1f}%', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def create_summary_table(self, threshold=0.10):
        """Create a pandas DataFrame with summary statistics"""
        data = {
            'Overall PCK': [self.metrics['overall'][threshold]],
        }

        # Add per-category stats
        for cat in sorted(self.metrics['per_category'].keys()):
            data[f'{cat}'] = [self.metrics['per_category'][cat][threshold]]

        df = pd.DataFrame(data)
        return df

    def export_to_csv(self, save_dir='./results'):
        """Export all metrics to CSV files"""
        import os
        os.makedirs(save_dir, exist_ok=True)

        # Overall PCK
        overall_df = pd.DataFrame([
            {'threshold': t, 'pck': self.metrics['overall'][t]}
            for t in sorted(self.metrics['overall'].keys())
        ])
        overall_df.to_csv(f'{save_dir}/overall_pck.csv', index=False)

        # Per-category PCK
        category_data = []
        for cat in sorted(self.metrics['per_category'].keys()):
            for t in sorted(self.metrics['overall'].keys()):
                category_data.append({
                    'category': cat,
                    'threshold': t,
                    'pck': self.metrics['per_category'][cat][t]
                })
        category_df = pd.DataFrame(category_data)
        category_df.to_csv(f'{save_dir}/per_category_pck.csv', index=False)

        # Per-keypoint PCK (now with category)
        keypoint_data = []
        for category in sorted(self.metrics['per_keypoint'].keys()):
            for kp_id in sorted(self.metrics['per_keypoint'][category].keys()):
                for t in sorted(self.metrics['overall'].keys()):
                    keypoint_data.append({
                        'category': category,
                        'keypoint_id': kp_id,
                        'threshold': t,
                        'pck': self.metrics['per_keypoint'][category][kp_id][t]
                    })
        keypoint_df = pd.DataFrame(keypoint_data)
        keypoint_df.to_csv(f'{save_dir}/per_keypoint_pck.csv', index=False)

        # Per-image PCK
        image_data = []
        for idx in sorted(self.metrics['per_image'].keys()):
            for t in sorted(self.metrics['overall'].keys()):
                image_data.append({
                    'pair_idx': idx,
                    'threshold': t,
                    'pck': self.metrics['per_image'][idx][t]
                })
        image_df = pd.DataFrame(image_data)
        image_df.to_csv(f'{save_dir}/per_image_pck.csv', index=False)

        print(f"✅ Exported all metrics to {save_dir}/")

    def generate_report(self, save_dir='./results'):
        """Generate a complete visual report"""
        import os
        os.makedirs(save_dir, exist_ok=True)

        print("📊 Generating visual report...")

        self.plot_pck_curve(save_path=f'{save_dir}/pck_curve.png')
        self.plot_per_category(save_path=f'{save_dir}/per_category.png')
        self.plot_keypoint_difficulty(save_path=f'{save_dir}/keypoint_difficulty.png')
        self.plot_image_difficulty_distribution(save_path=f'{save_dir}/image_distribution.png')
        self.export_to_csv(save_dir=save_dir)

        print(f"✅ Report generated in {save_dir}/")

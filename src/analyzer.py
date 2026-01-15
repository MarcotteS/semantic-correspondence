import matplotlib.pyplot as plt
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
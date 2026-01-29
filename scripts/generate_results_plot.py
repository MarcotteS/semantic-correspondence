"""Generate comparison bar chart for PCK@0.05 and PCK@0.10."""
import matplotlib.pyplot as plt
import numpy as np

models = ['DINOv2', 'DINOv3', 'SAM']
methods = ['Baseline', 'Baseline+WSA', 'Fine-tuned', 'Fine-tuned+WSA']

# Updated data with correct DINOv3 FT results
data_005 = {
    'DINOv2': [36.92, 39.33, 50.55, 52.36],
    'DINOv3': [35.41, 33.44, 57.80, 61.51],
    'SAM': [13.37, 13.02, 29.69, 31.09],
}
data_010 = {
    'DINOv2': [53.93, 55.85, 64.00, 64.48],
    'DINOv3': [52.77, 54.05, 74.36, 76.79],
    'SAM': [22.67, 23.23, 42.63, 44.54],
}

colors = ['#7faec9', '#a8d5e5', '#d97b53', '#f5a67a']

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

for ax, (data, title) in zip(axes, [(data_005, 'PCK@0.05'), (data_010, 'PCK@0.10')]):
    x = np.arange(len(models))
    width = 0.2
    
    for i, method in enumerate(methods):
        vals = [data[m][i] for m in models]
        bars = ax.bar(x + i*width, vals, width, label=method, color=colors[i], edgecolor='k', lw=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8, f'{v:.1f}', 
                    ha='center', va='bottom', fontsize=7)
    
    ax.set_ylabel(f'{title} (%)', fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xticks(x + 1.5*width)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 85)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

axes[1].legend(loc='upper right', fontsize=8, framealpha=0.9)
plt.tight_layout()
plt.savefig('results/pck_comparison.pdf', dpi=300, bbox_inches='tight')
plt.savefig('results/pck_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: results/pck_comparison.pdf")

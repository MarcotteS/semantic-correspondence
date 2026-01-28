"""Script to plot SD per-category PCK against DINOv2's"""

import matplotlib.pyplot as plt
import numpy as np


category_data = {
    'aeroplane': {'sd': 55.85, 'dinov2': 69.65, 'type': 'articulated'}, 
    'bicycle': {'sd': 45.68, 'dinov2': 60.01, 'type': 'articulated'}, 
    'bird': {'sd': 68.97, 'dinov2': 84.22, 'type': 'articulated'}, 
    'boat': {'sd': 28.91, 'dinov2': 33.28, 'type': 'simple'}, 
    'bottle': {'sd': 40.84, 'dinov2': 41.45, 'type': 'simple'}, 
    'bus': {'sd': 42.59, 'dinov2': 50.09, 'type': 'simple'}, 
    'car': {'sd': 41.56, 'dinov2': 47.19, 'type': 'simple'}, 
    'cat': {'sd': 69.62, 'dinov2': 67.99, 'type': 'articulated'},         
    'chair': {'sd': 32.86, 'dinov2': 36.06, 'type': 'simple'}, 
    'cow': {'sd': 62.11, 'dinov2': 66.52, 'type': 'articulated'}, 
    'dog': {'sd': 48.29, 'dinov2': 65.06, 'type': 'articulated'}, 
    'horse': {'sd': 52.27, 'dinov2': 66.35, 'type': 'articulated'}, 
    'motorbike': {'sd': 40.48, 'dinov2': 56.38, 'type': 'articulated'}, 
    'person': {'sd': 39.71, 'dinov2': 66.06, 'type': 'articulated'}, 
    'pottedplant': {'sd': 48.54, 'dinov2': 31.58, 'type': 'simple'},
    'sheep': {'sd': 40.58, 'dinov2': 60.50, 'type': 'articulated'}, 
    'train': {'sd': 57.99, 'dinov2': 49.23, 'type': 'simple'}, 
    'tvmonitor': {'sd': 49.41, 'dinov2': 27.38, 'type': 'simple'}
}

articulated_data = {k: v for k, v in category_data.items() if v['type'] == 'articulated'}
simple_data = {k: v for k, v in category_data.items() if v['type'] == 'simple'}

fig, ax = plt.subplots(figsize=(8, 6))

articulated_x = [data['dinov2'] for data in articulated_data.values()]
articulated_y = [data['sd'] for data in articulated_data.values()]
ax.scatter(
    articulated_x, articulated_y,
    s=100, marker='o', alpha=0.7, 
    color='#e74c3c', edgecolors='black', linewidth=1,
    label='Articulated objects', zorder=3
)

simple_x = [data['dinov2'] for data in simple_data.values()]
simple_y = [data['sd'] for data in simple_data.values()]
ax.scatter(
    simple_x, simple_y,
    s=100, marker='s', alpha=0.7,
    color='#3498db', edgecolors='black', linewidth=1,
    label='Simple geometry', zorder=3
)

min_val = min(min(articulated_x + simple_x), min(articulated_y + simple_y))
max_val = max(max(articulated_x + simple_x), max(articulated_y + simple_y))
ax.plot([min_val, max_val], [min_val, max_val], 
        'k--', alpha=0.5, zorder=1, linewidth=1.5, label='Equal performance')


for cat in category_data:
    data = category_data[cat]
    ax.annotate(
        cat, 
        (data['dinov2'], data['sd']),
        xytext=(3,3),
        textcoords='offset points',
        fontsize=9,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                 edgecolor='gray', alpha=0.8),
        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', 
                      color='gray', lw=0.8)
    )


ax.set_xlabel('DINOv2 PCK@0.10 (%)', fontsize=12, fontweight='bold')
ax.set_ylabel('Stable Diffusion PCK@0.10 (%)', fontsize=12, fontweight='bold')
ax.set_title('Category-Specific Performance Comparison', fontsize=14, fontweight='bold', pad=15)

ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
ax.set_axisbelow(True)

ax.legend(loc='lower right', frameon=True, shadow=True, fontsize=10)

# Set equal aspect ratio and nice limits
ax.set_aspect('equal')
ax.set_xlim(20, 90)
ax.set_ylim(20, 90)

plt.tight_layout()
plt.show()

"""Script to plot SD per-category PCK against DINOv2's"""

import matplotlib.pyplot as plt
import numpy as np

categories = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 
            'chair', 'cow', 'dog', 'horse', 'motorbike', 'person', 'pottedplant',
            'sheep', 'train', 'tvmonitor']

category_data = {
    'aeroplane': {'sd': 55.85, 'dinov2': 69.65, 'type': 'rigid'}, 
    'bicycle': {'sd': 45.68, 'dinov2': 60.01, 'type': 'rigid'}, 
    'bird': {'sd': 68.97, 'dinov2': 84.22, 'type': 'deformable'}, 
    'boat': {'sd': 28.91, 'dinov2': 33.28, 'type': 'rigid'}, 
    'bottle': {'sd': 40.84, 'dinov2': 41.45, 'type': 'rigid'}, 
    'bus': {'sd': 42.59, 'dinov2': 50.09, 'type': 'rigid'}, 
    'car': {'sd': 41.56, 'dinov2': 47.19, 'type': 'rigid'}, 
    'cat': {'sd': 69.62, 'dinov2': 67.99, 'type': 'deformable'},         
    'chair': {'sd': 32.86, 'dinov2': 36.06, 'type': 'rigid'}, 
    'cow': {'sd': 62.11, 'dinov2': 66.52, 'type': 'deformable'}, 
    'dog': {'sd': 48.29, 'dinov2': 65.06, 'type': 'deformable'}, 
    'horse': {'sd': 52.27, 'dinov2': 66.35, 'type': 'deformable'}, 
    'motorbike': {'sd': 40.48, 'dinov2': 56.38, 'type': 'rigid'}, 
    'person': {'sd': 39.71, 'dinov2': 66.06, 'type': 'deformable'}, 
    'pottedplant': {'sd': 48.54, 'dinov2': 31.58, 'type': 'rigid'},
    'sheep': {'sd': 40.58, 'dinov2': 60.50, 'type': 'deformable'}, 
    'train': {'sd': 57.99, 'dinov2': 49.23, 'type': 'rigid'}, 
    'tvmonitor': {'sd': 49.41, 'dinov2': 27.38, 'type': 'rigid'}
}

deformable_data = {k: v for k, v in category_data.items() if v['type'] == 'deformable'}
rigid_data = {k: v for k, v in category_data.items() if v['type'] == 'rigid'}

fig, ax = plt.subplots(figsize=(8, 6))

deformable_x = [data['dinov2'] for data in deformable_data.values()]
deformable_y = [data['sd'] for data in deformable_data.values()]
ax.scatter(
    deformable_x, deformable_y,
    s=100, marker='o', alpha=0.7, 
    color='#e74c3c', edgecolors='black', linewidth=1,
    label='Deformable (animals, person)', zorder=3
)

rigid_x = [data['dinov2'] for data in rigid_data.values()]
rigid_y = [data['sd'] for data in rigid_data.values()]
ax.scatter(
    rigid_x, rigid_y,
    s=100, marker='s', alpha=0.7,
    color='#3498db', edgecolors='black', linewidth=1,
    label='Rigid (objects, vehicles)', zorder=3
)

min_val = min(min(deformable_x + rigid_x), min(deformable_y + rigid_y))
max_val = max(max(deformable_x + rigid_x), max(deformable_y + rigid_y))
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

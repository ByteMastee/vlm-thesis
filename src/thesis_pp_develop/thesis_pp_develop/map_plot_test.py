#!/usr/bin/env python3

import json
import numpy as np
import matplotlib.pyplot as plt

OBJECT_STACK_PATH = '/root/UVC_ws/vf_robot_model_ros2/step7_detections/object_stack2.json'
OUTPUT_PATH = '/root/UVC_ws/vf_robot_model_ros2/step7_detections/map_plot_2.png'

# Ground truth from Gazebo world file
ground_truth = {
    'chair_1': (-3.0,  2.0),
    'chair_2': (-3.5, -2.5),
    'couch':   ( 3.5,  0.0),
    'table':   ( 2.0,  2.5)
}

with open(OBJECT_STACK_PATH, 'r') as f:
    object_stack = json.load(f)

plt.figure(figsize=(10, 10))

# Plot ground truth
for label, (gx, gy) in ground_truth.items():
    plt.plot(gx, gy, 'g^', markersize=12)
    plt.annotate(f'GT: {label}\n({gx},{gy})', (gx, gy),
                 textcoords='offset points', xytext=(8, 8),
                 fontsize=9, color='green')

# Plot detected object stack
colors = ['red', 'orange', 'purple', 'cyan', 'magenta']
for i, (label, data) in enumerate(object_stack.items()):
    ox = data['x']
    oy = data['y']
    color = colors[i % len(colors)]
    plt.plot(ox, oy, '*', markersize=15, color=color)
    plt.annotate(f'Det: {label}\n({ox:.2f},{oy:.2f})', (ox, oy),
                 textcoords='offset points', xytext=(8, -18),
                 fontsize=9, color=color)

    # Draw error line to nearest GT
    best_gt_label = None
    best_dist = float('inf')
    for gt_label, (gx, gy) in ground_truth.items():
        dist = np.sqrt((ox - gx)**2 + (oy - gy)**2)
        if dist < best_dist:
            best_dist = dist
            best_gt_label = gt_label
            best_gx, best_gy = gx, gy

    plt.plot([ox, best_gx], [oy, best_gy], '--', color=color, linewidth=1.0)
    plt.text((ox + best_gx) / 2, (oy + best_gy) / 2,
             f'{best_dist:.2f}m', fontsize=8, color=color)

plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.title('Semantic Map — Detected vs Ground Truth')
plt.legend(handles=[
    plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='green',
               markersize=10, label='Ground Truth'),
    plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='red',
               markersize=10, label='Detected')
])
plt.grid(True)
plt.axis('equal')
plt.savefig(OUTPUT_PATH, dpi=150)
plt.show()
print(f'Plot saved to: {OUTPUT_PATH}')
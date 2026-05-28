"""
2D Top-Down Map Visualizer — Real-World Environment
=====================================================
Generates a white background 2D map plot showing:
- Robot trajectory
- Detected object positions with VLM labels only
- No GT (not available for real-world)

Works for YOLO+VLM and ViT+VLM pipelines.
"""

import os
import json
import matplotlib.pyplot as plt

# =============================================================================
# USER INPUTS — change these for every run
# =============================================================================

ROBOT_PATH_FILE   = "/root/UVC_ws/vf_robot_model_ros2/Final_Output/RW_FinalOutput/VITp1_4/VITp1_4_vit_robot_path.json"
OBJECT_STACK_FILE = "/root/UVC_ws/vf_robot_model_ros2/Final_Output/RW_FinalOutput/VITp1_4/VITp1_4_vit_vlm_object_stack.json"

RUN_NAME          = "RW_ViT_VLM_P1"   # used in output filename and plot title
LABEL_TYPE        = "vlm_label"             # "vlm_label" for both YOLO+VLM and ViT+VLM

# =============================================================================
# FIXED — do not change
# =============================================================================

OUTPUT_DIR = "/root/UVC_ws/vf_robot_model_ros2/Final_Output/MapPlots"

# =============================================================================


def load_robot_path(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data['x'], data['y']


def load_object_stack(path, label_type):
    with open(path, 'r') as f:
        data = json.load(f)
    objects = {}
    for key, val in data.items():
        objects[key] = {
            'x':     float(val['x']),
            'y':     float(val['y']),
            'label': val.get(label_type, key)
        }
    return objects


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    path_x, path_y = load_robot_path(ROBOT_PATH_FILE)
    objects        = load_object_stack(OBJECT_STACK_FILE, LABEL_TYPE)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    # ── Robot path ────────────────────────────────────────────────────────────
    ax.plot(path_y, path_x,
            color='#888888', linewidth=1.2,
            linestyle='--', zorder=1, label='Robot Path')

    ax.scatter(path_y[0], path_x[0],
               c='black', s=80, zorder=5, marker='o')
    ax.scatter(path_y[-1], path_x[-1],
               c='black', s=80, zorder=5, marker='s')
    ax.annotate('Start', (path_y[0], path_x[0]),
                textcoords='offset points', xytext=(6, 6),
                fontsize=8, color='black')
    ax.annotate('End', (path_y[-1], path_x[-1]),
                textcoords='offset points', xytext=(6, 6),
                fontsize=8, color='black')

    # ── Detected objects ──────────────────────────────────────────────────────
    for key, val in objects.items():
        ax.scatter(val['y'], val['x'],
                   c='#F44336', s=160, zorder=4,
                   marker='o', edgecolors='black', linewidths=0.8)
        ax.annotate(val['label'], (val['y'], val['x']),
                    textcoords='offset points', xytext=(8, -14),
                    fontsize=8.5, color='#F44336', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2',
                              facecolor='white', edgecolor='#F44336',
                              alpha=0.8))

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_elements = [
        plt.Line2D([0], [0], color='#888888', linewidth=1.2,
                   linestyle='--', label='Robot Path'),
        plt.scatter([], [], c='#F44336', s=100, marker='o',
                    edgecolors='black', linewidths=0.8, label='Detected Object'),
    ]
    ax.legend(handles=legend_elements, loc='lower right',
              fontsize=9, framealpha=0.9)

    # ── Axes formatting ───────────────────────────────────────────────────────
    ax.set_xlabel('Y (metres)', fontsize=11)
    ax.set_ylabel('X (metres)', fontsize=11)
    ax.set_title(f'Semantic Map — {RUN_NAME}', fontsize=13, fontweight='bold')
    ax.set_aspect('equal')
    ax.invert_xaxis()

    ax.grid(True, linestyle='--', alpha=0.4, color='gray')
    ax.axhline(0, color='black', linewidth=0.5, alpha=0.3)
    ax.axvline(0, color='black', linewidth=0.5, alpha=0.3)

    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, f'{RUN_NAME}_map.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'Map saved to: {output_path}')


if __name__ == '__main__':
    main()


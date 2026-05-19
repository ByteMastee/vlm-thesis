"""
Aggregated Statistical Analysis Plot for all SIM Environments
===============================================================
Single combined figure with 2 subplots:
1. Mean +- StD of SPE — grouped by environment, 3 pipelines per group
2. Mean +- StD of F1-Score — grouped by environment, 3 pipelines per group
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# =============================================================================
# USER INPUTS — update paths if needed
# =============================================================================

STAT_FILES = {
    'E1': "/root/UVC_ws/vf_robot_model_ros2/Final_Output/NewStat_Analysis/E1_statistical_analysis.json",
    'E2': "/root/UVC_ws/vf_robot_model_ros2/Final_Output/NewStat_Analysis/E2_statistical_analysis.json",
    'E3': "/root/UVC_ws/vf_robot_model_ros2/Final_Output/NewStat_Analysis/E3_statistical_analysis.json",
    'E4': "/root/UVC_ws/vf_robot_model_ros2/Final_Output/NewStat_Analysis/E4_statistical_analysis.json",
    'E5': "/root/UVC_ws/vf_robot_model_ros2/Final_Output/NewStat_Analysis/E5_statistical_analysis.json",
}

# =============================================================================
# FIXED — do not change
# =============================================================================

OUTPUT_DIR  = "/root/UVC_ws/vf_robot_model_ros2/Final_Output/FinalStat_Analysis"
OUTPUT_FILE = "SIM_aggregated_stat_analysis.png"

PIPELINES = ['YOLO', 'YOLO_VLM', 'VIT_VLM']

PIPELINE_COLORS = {
    'YOLO':     '#2196F3',
    'YOLO_VLM': '#4CAF50',
    'VIT_VLM':  '#FF9800'
}

PIPELINE_LABELS = {
    'YOLO':     'YOLO',
    'YOLO_VLM': 'YOLO + VLM',
    'VIT_VLM':  'ViT + VLM'
}

# =============================================================================


def load_stats(stat_files):
    """Load all environment statistical analysis JSONs."""
    data = {}
    for env, path in stat_files.items():
        with open(path, 'r') as f:
            data[env] = json.load(f)
    return data


def extract_values(data, metric_mean, metric_std):
    """
    Extract mean and std per pipeline per environment.
    Returns:
        means: dict {pipeline: [e1, e2, e3, e4, e5]}
        stds:  dict {pipeline: [e1, e2, e3, e4, e5]}
    """
    envs     = list(data.keys())
    means    = {p: [] for p in PIPELINES}
    stds     = {p: [] for p in PIPELINES}

    for env in envs:
        env_data = data[env]
        for p in PIPELINES:
            if p in env_data and env_data[p].get(metric_mean) is not None:
                means[p].append(env_data[p][metric_mean])
                stds[p].append(env_data[p][metric_std])
            else:
                means[p].append(0.0)
                stds[p].append(0.0)

    return means, stds


def plot_grouped_bars(ax, means, stds, envs, title, ylabel):
    n_envs     = len(envs)
    n_pipelines = len(PIPELINES)
    bar_width  = 0.25
    x          = np.arange(n_envs)

    for i, pipeline in enumerate(PIPELINES):
        offset = (i - 1) * bar_width
        bars = ax.bar(
            x + offset,
            means[pipeline],
            width=bar_width,
            yerr=stds[pipeline],
            capsize=4,
            color=PIPELINE_COLORS[pipeline],
            edgecolor='black',
            linewidth=0.7,
            label=PIPELINE_LABELS[pipeline]
        )

        # Value labels on bars
        for bar, mean in zip(bars, means[pipeline]):
            if mean > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 0.5,
                    f'{mean:.2f}',
                    ha='center', va='center',
                    fontsize=7, color='white', fontweight='bold'
                )

    ax.set_title(title, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(envs, fontsize=10)
    ax.set_xlabel('Environment', fontsize=10)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    data = load_stats(STAT_FILES)
    envs = list(data.keys())

    spe_means, spe_stds = extract_values(data, 'SPE_mean', 'SPE_std')
    f1_means,  f1_stds  = extract_values(data, 'F1_mean',  'F1_std')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Aggregated Statistical Analysis — SIM Environments',
                 fontsize=14, fontweight='bold')

    plot_grouped_bars(ax1, spe_means, spe_stds, envs,
                      title='Mean ± StD of SPE',
                      ylabel='SPE (metres)')

    plot_grouped_bars(ax2, f1_means, f1_stds, envs,
                      title='Mean ± StD of F1-Score',
                      ylabel='F1-Score')

    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Aggregated graph saved to: {output_path}')


if __name__ == '__main__':
    main()
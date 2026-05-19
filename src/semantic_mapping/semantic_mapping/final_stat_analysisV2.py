"""
Aggregated Statistical Analysis Plot for all SIM Environments
===============================================================
Single combined figure with 2 subplots:
1. Mean +- StD of SPE — grouped by pipeline, 5 environment bars per group
2. Mean +- StD of F1-Score — grouped by pipeline, 5 environment bars per group
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt

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
OUTPUT_FILE = "SIM_aggregated_stat_analysisV2.png"

PIPELINES = ['YOLO', 'YOLO_VLM', 'VIT_VLM']

PIPELINE_LABELS = {
    'YOLO':     'YOLO',
    'YOLO_VLM': 'YOLO + VLM',
    'VIT_VLM':  'ViT + VLM'
}

ENV_COLORS = {
    'E1': '#1f77b4',
    'E2': '#ff7f0e',
    'E3': '#2ca02c',
    'E4': '#d62728',
    'E5': '#9467bd'
}

# =============================================================================


def load_stats(stat_files):
    data = {}
    for env, path in stat_files.items():
        with open(path, 'r') as f:
            data[env] = json.load(f)
    return data


def extract_values(data, metric_mean, metric_std):
    """
    Returns:
        means: dict {pipeline: {env: value}}
        stds:  dict {pipeline: {env: value}}
    """
    envs  = list(data.keys())
    means = {p: {} for p in PIPELINES}
    stds  = {p: {} for p in PIPELINES}

    for env in envs:
        env_data = data[env]
        for p in PIPELINES:
            if p in env_data and env_data[p].get(metric_mean) is not None:
                means[p][env] = env_data[p][metric_mean]
                stds[p][env]  = env_data[p][metric_std]
            else:
                means[p][env] = 0.0
                stds[p][env]  = 0.0

    return means, stds


def plot_grouped_bars(ax, means, stds, envs, title, ylabel):
    n_pipelines = len(PIPELINES)
    n_envs      = len(envs)
    bar_width   = 0.15
    x           = np.arange(n_pipelines)

    for j, env in enumerate(envs):
        offset     = (j - (n_envs - 1) / 2) * bar_width
        env_means  = [means[p][env] for p in PIPELINES]
        env_stds   = [stds[p][env]  for p in PIPELINES]

        bars = ax.bar(
            x + offset,
            env_means,
            width=bar_width,
            yerr=env_stds,
            capsize=3,
            color=ENV_COLORS[env],
            edgecolor='black',
            linewidth=0.6,
            label=env
        )

        # Value labels on bars
        for bar, mean in zip(bars, env_means):
            if mean > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 0.5,
                    f'{mean:.2f}',
                    ha='center', va='center',
                    fontsize=6.5, color='white', fontweight='bold',
                    rotation=90
                )

    ax.set_title(title, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([PIPELINE_LABELS[p] for p in PIPELINES], fontsize=10)
    ax.set_xlabel('Pipeline', fontsize=10)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    ax.legend(title='Environment', fontsize=9, title_fontsize=9)


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
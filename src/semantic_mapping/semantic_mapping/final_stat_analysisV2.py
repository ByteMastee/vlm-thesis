"""
Aggregated Statistical Analysis Plot for all SIM Environments
===============================================================
Saves two separate PNG files:
1. SIM_aggregated_SPE.png     — Mean +- StD of SPE
2. SIM_aggregated_F1.png      — Mean +- StD of F1-Score
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

OUTPUT_DIR    = "/root/UVC_ws/vf_robot_model_ros2/Final_Output/Thesis_Figures"
OUTPUT_SPE    = "SIM_aggregated_SPE.png"
OUTPUT_F1     = "SIM_aggregated_F1.png"

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


def plot_single(means, stds, envs, title, ylabel, output_path, arrow):
    n_pipelines = len(PIPELINES)
    n_envs      = len(envs)
    bar_width   = 0.14
    x           = np.arange(n_pipelines)

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(f'{title}  {arrow}', fontsize=14, fontweight='bold')

    for j, env in enumerate(envs):
        offset    = (j - (n_envs - 1) / 2) * bar_width
        env_means = [means[p][env] for p in PIPELINES]
        env_stds  = [stds[p][env]  for p in PIPELINES]

        bars = ax.bar(
            x + offset,
            env_means,
            width=bar_width,
            yerr=env_stds,
            capsize=4,
            color=ENV_COLORS[env],
            edgecolor='black',
            linewidth=0.7,
            label=env
        )

        # Value labels above bars for better visibility
        for bar, mean, std in zip(bars, env_means, env_stds):
            if mean > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + std + (max(
                        [means[p][e] for p in PIPELINES for e in envs]
                    ) * 0.01),
                    f'{mean:.2f}',
                    ha='center', va='bottom',
                    fontsize=7.5, color='black', fontweight='bold',
                    rotation=90
                )

    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xlabel('Pipeline', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([PIPELINE_LABELS[p] for p in PIPELINES], fontsize=11)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    ax.legend(title='Environment', fontsize=9, title_fontsize=9,
              loc='upper right')

    # Add extra headroom for labels
    current_ylim = ax.get_ylim()
    ax.set_ylim(0, current_ylim[1] * 1.25)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {output_path}')


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    data = load_stats(STAT_FILES)
    envs = list(data.keys())

    spe_means, spe_stds = extract_values(data, 'SPE_mean', 'SPE_std')
    f1_means,  f1_stds  = extract_values(data, 'F1_mean',  'F1_std')

    plot_single(
        spe_means, spe_stds, envs,
        title='Mean ± StD of SPE',
        ylabel='SPE (metres)',
        output_path=os.path.join(OUTPUT_DIR, OUTPUT_SPE),
        arrow='↓'
    )

    plot_single(
        f1_means, f1_stds, envs,
        title='Mean ± StD of F1-Score',
        ylabel='F1-Score',
        output_path=os.path.join(OUTPUT_DIR, OUTPUT_F1),
        arrow='↑'
    )


if __name__ == '__main__':
    main()
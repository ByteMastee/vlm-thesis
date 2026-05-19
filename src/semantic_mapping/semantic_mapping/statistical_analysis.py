"""
Statistical Analysis Metric
==============================
Computed across multiple runs (trajectories) of the same environment.

1. Mean +- Standard Deviation of SPE
   - Measures position stability of each pipeline across trajectories

2. Mean +- Standard Deviation of F1-Score
   - Measures detection consistency of each pipeline across trajectories

Input: SPE and F1 result JSON files from individual runs of the same environment.
       Load as many run files as needed (2 runs per sim environment).

Note: Real-world environment has only 1 trajectory — statistical analysis
      is not applicable for it.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# USER INPUTS — change these for every environment
# =============================================================================

ENV_NAME = "E5"   # used in output filename

# SPE result JSON files for this environment (one per trajectory)
SPE_RUN_FILES = [
    "/root/UVC_ws/vf_robot_model_ros2/Final_Output/SPE_Metrics/E5_Path1_spe_results.json",
    "/root/UVC_ws/vf_robot_model_ros2/Final_Output/SPE_Metrics/E5_Path2_spe_results.json",
]

# F1 result JSON files for this environment (one per trajectory)
F1_RUN_FILES = [
    "/root/UVC_ws/vf_robot_model_ros2/Final_Output/F1_Metrics/E5_Path1_f1_results.json",
    "/root/UVC_ws/vf_robot_model_ros2/Final_Output/F1_Metrics/E5_Path2_f1_results.json",
]

# =============================================================================
# FIXED — do not change
# =============================================================================

OUTPUT_DIR = "/root/UVC_ws/vf_robot_model_ros2/Final_Output/NewStat_Analysis"

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


def load_spe_values(file_list):
    spe_per_pipeline = {}
    for path in file_list:
        with open(path, 'r') as f:
            data = json.load(f)
        for pipeline, values in data.items():
            if pipeline not in spe_per_pipeline:
                spe_per_pipeline[pipeline] = []
            spe = values.get('SPE')
            if spe is not None:
                spe_per_pipeline[pipeline].append(spe)
    return spe_per_pipeline


def load_f1_values(file_list):
    f1_per_pipeline = {}
    for path in file_list:
        with open(path, 'r') as f:
            data = json.load(f)
        for pipeline, values in data.items():
            if pipeline not in f1_per_pipeline:
                f1_per_pipeline[pipeline] = []
            f1 = values.get('F1')
            if f1 is not None:
                f1_per_pipeline[pipeline].append(f1)
    return f1_per_pipeline


def compute_stats(values_per_pipeline):
    """Compute Mean and StD for any metric per pipeline."""
    results = {}
    for pipeline, values in values_per_pipeline.items():
        arr  = np.array(values)
        mean = round(float(np.mean(arr)), 4)
        std  = round(float(np.std(arr)), 4)
        results[pipeline] = {
            'mean':   mean,
            'std':    std,
            'values': values,
            'N_runs': len(values)
        }
    return results


def generate_graphs(spe_stats, f1_stats, env_name, output_dir):
    pipelines = ['YOLO', 'YOLO_VLM', 'VIT_VLM']
    x         = np.arange(len(pipelines))
    bar_width = 0.5

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'Statistical Analysis — {env_name}', fontsize=14, fontweight='bold')

    # ── Subplot 1: Mean ± StD of SPE ──────────────────────────────────────────
    spe_means = [spe_stats[p]['mean'] if p in spe_stats else 0 for p in pipelines]
    spe_stds  = [spe_stats[p]['std']  if p in spe_stats else 0 for p in pipelines]
    colors    = [PIPELINE_COLORS[p] for p in pipelines]
    labels    = [PIPELINE_LABELS[p] for p in pipelines]

    bars1 = ax1.bar(x, spe_means, width=bar_width, color=colors,
                    yerr=spe_stds, capsize=6, edgecolor='black', linewidth=0.8)

    ax1.set_title('Mean ± StD of SPE', fontsize=12)
    ax1.set_ylabel('SPE (metres)', fontsize=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=10)
    ax1.set_ylim(0, (max(spe_means) + max(spe_stds)) * 1.3 if max(spe_means) > 0 else 1.0)
    ax1.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax1.set_axisbelow(True)

    for bar, mean in zip(bars1, spe_means):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() * 0.5,
                 f'{mean:.4f}', ha='center', va='bottom', fontsize=9)

    # ── Subplot 2: Mean ± StD of F1-Score ─────────────────────────────────────
    f1_means = [f1_stats[p]['mean'] if p in f1_stats else 0 for p in pipelines]
    f1_stds  = [f1_stats[p]['std']  if p in f1_stats else 0 for p in pipelines]

    bars2 = ax2.bar(x, f1_means, width=bar_width, color=colors,
                    yerr=f1_stds, capsize=6, edgecolor='black', linewidth=0.8)

    ax2.set_title('Mean ± StD of F1-Score', fontsize=12)
    ax2.set_ylabel('F1-Score', fontsize=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=10)
    ax2.set_ylim(0, (max(f1_means) + max(f1_stds)) * 1.3 if max(f1_means) > 0 else 1.0)
    ax2.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax2.set_axisbelow(True)

    for bar, mean in zip(bars2, f1_means):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() * 0.5,
                 f'{mean:.4f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    graph_path = os.path.join(output_dir, f'{env_name}_statistical_analysis.png')
    plt.savefig(graph_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Graph saved to: {graph_path}')


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    spe_per_pipeline = load_spe_values(SPE_RUN_FILES)
    f1_per_pipeline  = load_f1_values(F1_RUN_FILES)

    spe_stats = compute_stats(spe_per_pipeline)
    f1_stats  = compute_stats(f1_per_pipeline)

    all_pipelines = set(list(spe_stats.keys()) + list(f1_stats.keys()))
    results       = {}

    print(f'\n══ Statistical Analysis — {ENV_NAME} ══')

    for pipeline in sorted(all_pipelines):
        results[pipeline] = {}

        print(f'\n── {pipeline} ──')

        if pipeline in spe_stats:
            s = spe_stats[pipeline]
            results[pipeline]['SPE_mean']   = s['mean']
            results[pipeline]['SPE_std']    = s['std']
            results[pipeline]['SPE_values'] = s['values']
            results[pipeline]['N_runs']     = s['N_runs']
            print(f'  SPE Mean ± StD : {s["mean"]} ± {s["std"]} m')
            print(f'  SPE values     : {s["values"]}')

        if pipeline in f1_stats:
            f = f1_stats[pipeline]
            results[pipeline]['F1_mean']   = f['mean']
            results[pipeline]['F1_std']    = f['std']
            results[pipeline]['F1_values'] = f['values']
            print(f'  F1  Mean ± StD : {f["mean"]} ± {f["std"]}')
            print(f'  F1  values     : {f["values"]}')

    output_path = os.path.join(OUTPUT_DIR, f'{ENV_NAME}_statistical_analysis.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nStatistical analysis saved to: {output_path}')

    generate_graphs(spe_stats, f1_stats, ENV_NAME, OUTPUT_DIR)


if __name__ == '__main__':
    main()
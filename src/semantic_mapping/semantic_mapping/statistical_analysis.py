"""
Statistical Analysis Metric
==============================
Computed across multiple runs (trajectories) of the same environment.

1. Mean +- Standard Deviation of SPE
   - Measures position stability of each pipeline across trajectories

2. Coefficient of Variation (CV) of F1-Score
   CV = StD / Mean
   - Measures consistency of detection quality across trajectories

Input: SPE and F1 result JSON files from individual runs of the same environment.
       Load as many run files as needed (2 runs per sim environment).

Note: Real-world environment has only 1 trajectory — statistical analysis
      is not applicable for it.
"""

import os
import json
import numpy as np

# =============================================================================
# USER INPUTS — change these for every environment
# =============================================================================

ENV_NAME = "E1"   # used in output filename

# SPE result JSON files for this environment (one per trajectory)
SPE_RUN_FILES = [
    "/root/UVC_ws/vf_robot_model_ros2/Final_Output/QuantityMetrics/E1_Path1_spe_results.json",
    "/root/UVC_ws/vf_robot_model_ros2/Final_Output/QuantityMetrics/E1_Path2_spe_results.json",
]

# F1 result JSON files for this environment (one per trajectory)
F1_RUN_FILES = [
    "/root/UVC_ws/vf_robot_model_ros2/Final_Output/QuantityMetrics/E1_Path1_f1_results.json",
    "/root/UVC_ws/vf_robot_model_ros2/Final_Output/QuantityMetrics/E1_Path2_f1_results.json",
]

# =============================================================================
# FIXED — do not change
# =============================================================================

OUTPUT_DIR = "/root/UVC_ws/vf_robot_model_ros2/Final_Output/QuantityMetrics"

# =============================================================================


def load_spe_values(file_list):
    """
    Load SPE value per pipeline from each run file.
    Returns dict {pipeline: [spe_run1, spe_run2, ...]}
    """
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
    """
    Load F1 value per pipeline from each run file.
    Returns dict {pipeline: [f1_run1, f1_run2, ...]}
    """
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


def compute_spe_stats(spe_per_pipeline):
    """
    Compute Mean and StD of SPE per pipeline.
    Returns dict {pipeline: {mean, std, values}}
    """
    results = {}
    for pipeline, values in spe_per_pipeline.items():
        arr  = np.array(values)
        mean = round(float(np.mean(arr)), 4)
        std  = round(float(np.std(arr)), 4)
        results[pipeline] = {
            'mean_SPE': mean,
            'std_SPE':  std,
            'values':   values,
            'N_runs':   len(values)
        }
    return results


def compute_f1_cv(f1_per_pipeline):
    """
    Compute CV of F1-Score per pipeline.
    CV = StD / Mean
    Returns dict {pipeline: {mean, std, cv, values}}
    """
    results = {}
    for pipeline, values in f1_per_pipeline.items():
        arr  = np.array(values)
        mean = round(float(np.mean(arr)), 4)
        std  = round(float(np.std(arr)), 4)
        cv   = round(float(std / mean), 4) if mean > 0 else None
        results[pipeline] = {
            'mean_F1': mean,
            'std_F1':  std,
            'CV_F1':   cv,
            'values':  values,
            'N_runs':  len(values)
        }
    return results


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    spe_per_pipeline = load_spe_values(SPE_RUN_FILES)
    f1_per_pipeline  = load_f1_values(F1_RUN_FILES)

    spe_stats = compute_spe_stats(spe_per_pipeline)
    f1_stats  = compute_f1_cv(f1_per_pipeline)

    # Merge into combined results per pipeline
    all_pipelines = set(list(spe_stats.keys()) + list(f1_stats.keys()))
    results       = {}

    print(f'\n══ Statistical Analysis — {ENV_NAME} ══')

    for pipeline in sorted(all_pipelines):
        results[pipeline] = {}

        print(f'\n── {pipeline} ──')

        if pipeline in spe_stats:
            s = spe_stats[pipeline]
            results[pipeline]['SPE_mean']   = s['mean_SPE']
            results[pipeline]['SPE_std']    = s['std_SPE']
            results[pipeline]['SPE_values'] = s['values']
            results[pipeline]['N_runs']     = s['N_runs']
            print(f'  SPE Mean ± StD : {s["mean_SPE"]} ± {s["std_SPE"]} m')
            print(f'  SPE values     : {s["values"]}')

        if pipeline in f1_stats:
            f = f1_stats[pipeline]
            results[pipeline]['F1_mean']   = f['mean_F1']
            results[pipeline]['F1_std']    = f['std_F1']
            results[pipeline]['F1_CV']     = f['CV_F1']
            results[pipeline]['F1_values'] = f['values']
            print(f'  F1  Mean ± StD : {f["mean_F1"]} ± {f["std_F1"]}')
            print(f'  F1  CV         : {f["CV_F1"]}')
            print(f'  F1  values     : {f["values"]}')

    output_path = os.path.join(OUTPUT_DIR, f'{ENV_NAME}_statistical_analysis.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nStatistical analysis saved to: {output_path}')


if __name__ == '__main__':
    main()
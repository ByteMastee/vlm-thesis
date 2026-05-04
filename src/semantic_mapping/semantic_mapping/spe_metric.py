"""
Spatial Position Error (SPE) Metric
=====================================
SPE = (1/N) * sum(|| pos_detected - pos_GT ||_2)

- Matches each detected object to its closest GT object using
  Hungarian algorithm (optimal assignment by Euclidean distance).
- N = number of matched pairs.
- Unmatched GT objects (missed detections) are reported separately.
"""

import os
import json
import numpy as np
from scipy.optimize import linear_sum_assignment

# =============================================================================
# USER INPUTS — change these for every run
# =============================================================================

GT_STRING     = "chair_1:-3.0:2.0,chair_2:-3.5:-2.5,table:2.0:2.5,couch:3.5:0.0"

YOLO_PATH     = "/root/UVC_ws/vf_robot_model_ros2/Final_Output/PreFinal/E1_Path1_1/E1_Path1_1_object_stack.json"
YOLO_VLM_PATH = "/root/UVC_ws/vf_robot_model_ros2/Final_Output/PreFinal/E1_Path1_1/E1_Path1_1_vlm_object_stack.json"
VIT_VLM_PATH  = "/root/UVC_ws/vf_robot_model_ros2/Final_Output/ViT/Env1_Path1/Env1_Path1_vit_vlm_object_stack.json"

RUN_NAME      = "E1_Path1"   # used in output filename

# =============================================================================
# FIXED — do not change
# =============================================================================

OUTPUT_DIR    = "/root/UVC_ws/vf_robot_model_ros2/Final_Output/SPE_Metrics"

# =============================================================================


def parse_gt(gt_string):
    gt = {}
    for entry in gt_string.split(','):
        entry = entry.strip()
        parts = entry.split(':')
        label = parts[0].strip()
        x     = float(parts[1])
        y     = float(parts[2])
        gt[label] = (x, y)
    return gt


def load_object_stack(path):
    with open(path, 'r') as f:
        data = json.load(f)
    objects = {}
    for key, val in data.items():
        objects[key] = (float(val['x']), float(val['y']))
    return objects


def compute_spe(gt, predicted):
    gt_labels   = list(gt.keys())
    pred_labels = list(predicted.keys())

    if len(pred_labels) == 0:
        return None, [], gt_labels, []

    gt_positions   = np.array([gt[l] for l in gt_labels])
    pred_positions = np.array([predicted[l] for l in pred_labels])

    n_gt   = len(gt_labels)
    n_pred = len(pred_labels)
    cost   = np.zeros((n_gt, n_pred))

    for i in range(n_gt):
        for j in range(n_pred):
            cost[i, j] = np.linalg.norm(
                np.array(gt_positions[i]) - np.array(pred_positions[j])
            )

    row_ind, col_ind = linear_sum_assignment(cost)

    matched_pairs    = []
    matched_gt_idx   = set()
    matched_pred_idx = set()

    for r, c in zip(row_ind, col_ind):
        dist = cost[r, c]
        matched_pairs.append((pred_labels[c], gt_labels[r], round(dist, 4)))
        matched_gt_idx.add(r)
        matched_pred_idx.add(c)

    distances      = [pair[2] for pair in matched_pairs]
    spe            = round(float(np.mean(distances)), 4) if distances else None
    unmatched_gt   = [gt_labels[i]   for i in range(n_gt)   if i not in matched_gt_idx]
    unmatched_pred = [pred_labels[j] for j in range(n_pred) if j not in matched_pred_idx]

    return spe, matched_pairs, unmatched_gt, unmatched_pred


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    gt       = parse_gt(GT_STRING)
    yolo     = load_object_stack(YOLO_PATH)
    yolo_vlm = load_object_stack(YOLO_VLM_PATH)
    vit_vlm  = load_object_stack(VIT_VLM_PATH)

    pipelines = {
        'YOLO':     yolo,
        'YOLO_VLM': yolo_vlm,
        'VIT_VLM':  vit_vlm
    }

    results = {}

    for name, predicted in pipelines.items():
        spe, matched, unmatched_gt, unmatched_pred = compute_spe(gt, predicted)

        results[name] = {
            'SPE':            spe,
            'matched_pairs':  matched,
            'unmatched_gt':   unmatched_gt,
            'unmatched_pred': unmatched_pred,
            'N_gt':           len(gt),
            'N_detected':     len(predicted),
            'N_matched':      len(matched)
        }

        print(f'\n── {name} ──')
        print(f'  SPE        : {spe} m')
        print(f'  GT objects : {len(gt)}')
        print(f'  Detected   : {len(predicted)}')
        print(f'  Matched    : {len(matched)}')
        for pred_l, gt_l, dist in matched:
            print(f'    pred="{pred_l}" <-> gt="{gt_l}"  dist={dist} m')
        if unmatched_gt:
            print(f'  Unmatched GT   : {unmatched_gt}')
        if unmatched_pred:
            print(f'  Unmatched pred : {unmatched_pred}')

    output_path = os.path.join(OUTPUT_DIR, f'{RUN_NAME}_spe_results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSPE results saved to: {output_path}')


if __name__ == '__main__':
    main()
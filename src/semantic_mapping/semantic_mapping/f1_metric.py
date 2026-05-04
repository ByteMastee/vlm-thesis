"""
Object Detection F1-Score Metric
===================================
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2 * Precision * Recall / (Precision + Recall)

Definitions:
- TP: position-matched pair where label similarity >= threshold
- FP: detection with no GT match OR label similarity < threshold
- FN: GT object with no detection OR matched but label similarity < threshold

- Position matching uses Hungarian algorithm.
- Label matching uses cosine similarity with configurable threshold.
"""

import os
import json
import numpy as np
from scipy.optimize import linear_sum_assignment
from sentence_transformers import SentenceTransformer

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

OUTPUT_DIR    = "/root/UVC_ws/vf_robot_model_ros2/Final_Output/F1_Metrics"
MODEL_PATH    = "/root/all-MiniLM-L6-v2"
THRESHOLD     = 0.6

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


def load_object_stack_with_labels(path):
    with open(path, 'r') as f:
        data = json.load(f)
    objects = {}
    for key, val in data.items():
        label = val.get('vlm_label', key)
        objects[key] = {
            'x':     float(val['x']),
            'y':     float(val['y']),
            'label': label
        }
    return objects


def cosine_similarity(vec1, vec2):
    dot  = np.dot(vec1, vec2)
    norm = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    if norm == 0:
        return 0.0
    return float(dot / norm)


def compute_f1(gt, predicted, model):
    gt_labels  = list(gt.keys())
    pred_keys  = list(predicted.keys())

    if len(pred_keys) == 0:
        fn      = len(gt_labels)
        details = [{'gt_label': l, 'result': 'FN', 'reason': 'no detections'}
                   for l in gt_labels]
        return 0.0, 0.0, 0.0, 0, fn, fn, details

    gt_positions   = np.array([gt[l] for l in gt_labels])
    pred_positions = np.array([(predicted[k]['x'], predicted[k]['y']) for k in pred_keys])

    n_gt   = len(gt_labels)
    n_pred = len(pred_keys)
    cost   = np.zeros((n_gt, n_pred))

    for i in range(n_gt):
        for j in range(n_pred):
            cost[i, j] = np.linalg.norm(gt_positions[i] - pred_positions[j])

    row_ind, col_ind = linear_sum_assignment(cost)

    gt_embeddings   = {l: model.encode(l) for l in gt_labels}
    pred_embeddings = {k: model.encode(predicted[k]['label']) for k in pred_keys}

    matched_gt_idx   = set()
    matched_pred_idx = set()
    details          = []
    tp = fp = fn = 0

    for r, c in zip(row_ind, col_ind):
        gt_label   = gt_labels[r]
        pred_key   = pred_keys[c]
        pred_label = predicted[pred_key]['label']
        dist       = round(cost[r, c], 4)
        sim        = round(cosine_similarity(gt_embeddings[gt_label],
                                             pred_embeddings[pred_key]), 4)
        correct    = sim >= THRESHOLD

        if correct:
            tp += 1
            result = 'TP'
        else:
            fp += 1
            fn += 1
            result = 'FP+FN'

        matched_gt_idx.add(r)
        matched_pred_idx.add(c)

        details.append({
            'pred_key':   pred_key,
            'pred_label': pred_label,
            'gt_label':   gt_label,
            'distance_m': dist,
            'similarity': sim,
            'result':     result
        })

    for i in range(n_gt):
        if i not in matched_gt_idx:
            fn += 1
            details.append({
                'gt_label': gt_labels[i],
                'result':   'FN',
                'reason':   'no matched detection'
            })

    for j in range(n_pred):
        if j not in matched_pred_idx:
            fp += 1
            details.append({
                'pred_key':   pred_keys[j],
                'pred_label': predicted[pred_keys[j]]['label'],
                'result':     'FP',
                'reason':     'no matched GT'
            })

    precision = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0
    recall    = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0
    f1        = round(2 * precision * recall / (precision + recall), 4) \
                if (precision + recall) > 0 else 0.0

    return precision, recall, f1, tp, fp, fn, details


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f'Loading sentence-transformer model from: {MODEL_PATH}')
    model = SentenceTransformer(MODEL_PATH)

    gt = parse_gt(GT_STRING)

    pipelines = {
        'YOLO':     load_object_stack_with_labels(YOLO_PATH),
        'YOLO_VLM': load_object_stack_with_labels(YOLO_VLM_PATH),
        'VIT_VLM':  load_object_stack_with_labels(VIT_VLM_PATH)
    }

    results = {}

    for name, predicted in pipelines.items():
        precision, recall, f1, tp, fp, fn, details = compute_f1(gt, predicted, model)

        results[name] = {
            'Precision':  precision,
            'Recall':     recall,
            'F1':         f1,
            'TP':         tp,
            'FP':         fp,
            'FN':         fn,
            'threshold':  THRESHOLD,
            'N_gt':       len(gt),
            'N_detected': len(predicted),
            'details':    details
        }

        print(f'\n── {name} ──')
        print(f'  Precision : {precision}')
        print(f'  Recall    : {recall}')
        print(f'  F1        : {f1}')
        print(f'  TP={tp}  FP={fp}  FN={fn}')
        for d in details:
            if 'pred_label' in d:
                print(f'  [{d["result"]}] pred="{d["pred_label"]}" <-> gt="{d.get("gt_label","—")}"'
                      f'  sim={d.get("similarity","—")}  dist={d.get("distance_m","—")}m')
            else:
                print(f'  [{d["result"]}] gt="{d["gt_label"]}"  ({d.get("reason","")})')

    output_path = os.path.join(OUTPUT_DIR, f'{RUN_NAME}_f1_results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nF1 results saved to: {output_path}')


if __name__ == '__main__':
    main()
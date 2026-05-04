"""
Semantic Similarity Score (SSS) Metric
=========================================
SSS = cosine_similarity(embed(vlm_label), embed(gt_label))

- Computed only for YOLO_VLM and VIT_VLM pipelines.
- Position matching uses Hungarian algorithm (same as all other metrics).
- Raw cosine similarity score is reported per matched pair (no threshold).
- Final SSS = mean cosine similarity across all matched pairs.
- Unmatched GT objects (missed detections) contribute 0.0 to the mean.
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

YOLO_VLM_PATH = "/root/UVC_ws/vf_robot_model_ros2/Final_Output/PreFinal/E1_Path1_1/E1_Path1_1_vlm_object_stack.json"
VIT_VLM_PATH  = "/root/UVC_ws/vf_robot_model_ros2/Final_Output/ViT/Env1_Path1/Env1_Path1_vit_vlm_object_stack.json"

RUN_NAME      = "E1_Path1"   # used in output filename

# =============================================================================
# FIXED — do not change
# =============================================================================

OUTPUT_DIR    = "/root/UVC_ws/vf_robot_model_ros2/Final_Output/SSS_Metrics"
MODEL_PATH    = "/root/all-MiniLM-L6-v2"

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


def compute_sss(gt, predicted, model):
    """
    Compute SSS for one pipeline.

    SSS = mean cosine similarity across all GT objects.
    - Matched pairs: similarity between pred label and GT label embeddings.
    - Unmatched GT objects: similarity = 0.0 (missed detection penalty).

    Parameters:
        gt        : dict {label: (x, y)}
        predicted : dict {key: {x, y, label}}
        model     : SentenceTransformer model

    Returns:
        sss      : float — mean similarity across all GT objects
        details  : list of per-GT-object result dicts
    """
    gt_labels = list(gt.keys())
    pred_keys = list(predicted.keys())

    # Default: all GT unmatched with similarity 0.0
    details = {l: {
        'gt_label':   l,
        'pred_label': None,
        'similarity': 0.0,
        'matched':    False
    } for l in gt_labels}

    if len(pred_keys) == 0:
        sss = 0.0
        return sss, list(details.values())

    gt_positions   = np.array([gt[l] for l in gt_labels])
    pred_positions = np.array([(predicted[k]['x'], predicted[k]['y']) for k in pred_keys])

    n_gt   = len(gt_labels)
    n_pred = len(pred_keys)
    cost   = np.zeros((n_gt, n_pred))

    for i in range(n_gt):
        for j in range(n_pred):
            cost[i, j] = np.linalg.norm(gt_positions[i] - pred_positions[j])

    row_ind, col_ind = linear_sum_assignment(cost)

    gt_embeddings = {l: model.encode(l) for l in gt_labels}

    for r, c in zip(row_ind, col_ind):
        gt_label   = gt_labels[r]
        pred_key   = pred_keys[c]
        pred_label = predicted[pred_key]['label']
        dist       = round(cost[r, c], 4)

        pred_emb = model.encode(pred_label)
        gt_emb   = gt_embeddings[gt_label]
        sim      = round(cosine_similarity(pred_emb, gt_emb), 4)

        details[gt_label] = {
            'gt_label':   gt_label,
            'pred_key':   pred_key,
            'pred_label': pred_label,
            'distance_m': dist,
            'similarity': sim,
            'matched':    True
        }

    # Mean over all GT objects (unmatched contribute 0.0)
    similarities = [d['similarity'] for d in details.values()]
    sss          = round(float(np.mean(similarities)), 4)

    return sss, list(details.values())


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f'Loading sentence-transformer model from: {MODEL_PATH}')
    model = SentenceTransformer(MODEL_PATH)

    gt = parse_gt(GT_STRING)

    pipelines = {
        'YOLO_VLM': load_object_stack_with_labels(YOLO_VLM_PATH),
        'VIT_VLM':  load_object_stack_with_labels(VIT_VLM_PATH)
    }

    results = {}

    for name, predicted in pipelines.items():
        sss, details = compute_sss(gt, predicted, model)

        results[name] = {
            'SSS':        sss,
            'N_gt':       len(gt),
            'N_detected': len(predicted),
            'N_matched':  sum(1 for d in details if d['matched']),
            'details':    details
        }

        print(f'\n── {name} ──')
        print(f'  SSS       : {sss}')
        print(f'  GT objects: {len(gt)}')
        print(f'  Matched   : {sum(1 for d in details if d["matched"])}')
        for d in details:
            if d['matched']:
                print(f'  pred="{d["pred_label"]}" <-> gt="{d["gt_label"]}"'
                      f'  sim={d["similarity"]}  dist={d.get("distance_m","—")}m')
            else:
                print(f'  [UNMATCHED] gt="{d["gt_label"]}"  sim=0.0')

    output_path = os.path.join(OUTPUT_DIR, f'{RUN_NAME}_sss_results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSSS results saved to: {output_path}')


if __name__ == '__main__':
    main()
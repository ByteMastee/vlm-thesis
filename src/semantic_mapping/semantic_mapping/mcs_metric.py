"""
Map Completeness Score (MCS) Metric
======================================
MCS = Number of correctly mapped objects / Total objects in GT map

- An object is "correctly mapped" if:
    1. A detected object is position-matched to it (Hungarian algorithm)
    2. AND the label similarity >= threshold

- Separated from SLA: MCS measures coverage (did the pipeline find the
  object at all AND label it correctly), while SLA only measures label
  quality of matched objects.

Note:
- MCS denominator is always total GT objects.
- An unmatched GT object (missed detection) counts as not correctly mapped.
- A matched but wrongly labelled object also counts as not correctly mapped.
"""

import os
import json
import numpy as np
from scipy.optimize import linear_sum_assignment
from sentence_transformers import SentenceTransformer

# =============================================================================
# USER INPUTS — change these for every run
# =============================================================================

GT_STRING     = "reception_table:-5.0:5.5,reception_chair:-5.0:6.6,dustbin:0.0:6.0,chair_1:4.5:5.5,chair_2:5.7:5.5,chair_3:6.9:5.5,chair_4:8.1:5.5,potted_plant:6.0:-4.5,office_desk:7.5:2.0,office_chair:7.5:0.9,bookshelf:9.7:0.5,filing_cabinet:4.2:-1.8,operation_table:-6.5:0.5,instrument_trolley:-5.5:-0.5,medical_monitor:-7.8:1.5,iv_stand:-7.5:-0.5"

YOLO_PATH     = "/root/UVC_ws/vf_robot_model_ros2/Final_Output/PreFinal/E5_Path2_1/E5_Path2_1_object_stack.json"
YOLO_VLM_PATH = "/root/UVC_ws/vf_robot_model_ros2/Final_Output/PreFinal/E5_Path2_1/E5_Path2_1_vlm_object_stack.json"
VIT_VLM_PATH  = "/root/UVC_ws/vf_robot_model_ros2/Final_Output/ViT/Env5_Path2/Env5_Path2_vit_vlm_object_stack.json"

RUN_NAME      = "E5_Path2"   # used in output filename

# =============================================================================
# FIXED — do not change
# =============================================================================

OUTPUT_DIR    = "/root/UVC_ws/vf_robot_model_ros2/Final_Output/MCS_Metrics"
MODEL_PATH    = "/root/all-MiniLM-L6-v2"
THRESHOLD     = 0.6

# =============================================================================


def parse_gt(gt_string):
    gt = {}
    for entry in gt_string.split(','):
        entry  = entry.strip()
        parts  = entry.split(':')
        label  = parts[0].strip()
        x      = float(parts[1])
        y      = float(parts[2])
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


def compute_mcs(gt, predicted, model):
    """
    Compute MCS for one pipeline.

    MCS = correctly mapped objects / total GT objects

    An object is correctly mapped if:
    - It is position-matched to a GT object (Hungarian)
    - AND label cosine similarity >= threshold

    Parameters:
        gt        : dict {label: (x, y)}
        predicted : dict {key: {x, y, label}}
        model     : SentenceTransformer model

    Returns:
        mcs      : float
        details  : list of per-GT-object result dicts
    """
    gt_labels = list(gt.keys())
    pred_keys = list(predicted.keys())

    # Build result entry for every GT object (default: not mapped)
    details = {l: {
        'gt_label':          l,
        'correctly_mapped':  False,
        'reason':            'no detection'
    } for l in gt_labels}

    if len(pred_keys) == 0:
        return 0.0, list(details.values())

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
        correct  = sim >= THRESHOLD

        details[gt_label] = {
            'gt_label':         gt_label,
            'pred_key':         pred_key,
            'pred_label':       pred_label,
            'distance_m':       dist,
            'similarity':       sim,
            'correctly_mapped': correct,
            'reason':           'matched and labelled correctly' if correct
                                else 'matched but label wrong'
        }

    correctly_mapped = sum(1 for d in details.values() if d['correctly_mapped'])
    mcs              = round(correctly_mapped / len(gt_labels), 4)

    return mcs, list(details.values())


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
        mcs, details = compute_mcs(gt, predicted, model)

        results[name] = {
            'MCS':              mcs,
            'threshold':        THRESHOLD,
            'N_gt':             len(gt),
            'N_detected':       len(predicted),
            'N_correctly_mapped': sum(1 for d in details if d['correctly_mapped']),
            'details':          details
        }

        print(f'\n── {name} ──')
        print(f'  MCS       : {mcs}  '
              f'({sum(1 for d in details if d["correctly_mapped"])}/{len(gt)} correctly mapped)')
        for d in details:
            status = 'MAPPED' if d['correctly_mapped'] else 'NOT MAPPED'
            if 'pred_label' in d:
                print(f'  [{status}] gt="{d["gt_label"]}" <- pred="{d["pred_label"]}"'
                      f'  sim={d["similarity"]}  dist={d["distance_m"]}m')
            else:
                print(f'  [{status}] gt="{d["gt_label"]}"  ({d["reason"]})')

    output_path = os.path.join(OUTPUT_DIR, f'{RUN_NAME}_mcs_results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nMCS results saved to: {output_path}')


if __name__ == '__main__':
    main()
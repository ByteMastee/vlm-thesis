#!/usr/bin/env python3
"""
F1-Score Evaluation Script for Real-World Semantic Mapping
Compares YOLO+VLM and ViT+VLM pipeline detections against GT objects.

Usage:
    python3 evaluate_f1.py \
        --vlm_stack  <path_to_vlm_object_stack.json> \
        --run_name   <run_name> \
        --output_dir <path_to_save_results>

Configuration:
    Edit GT_OBJECTS and NOISE_KEYWORDS below before running.

Definitions:
    TP = detected object whose label matches a GT object label
    FP = detected object whose label does not match any GT object
    FN = GT object not detected
    Precision = TP / (TP + FP)
    Recall    = TP / (TP + FN)
    F1        = 2 * Precision * Recall / (Precision + Recall)

MCS (Map Completeness Score):
    MCS = TP / Total GT objects
"""

import os
import json
import argparse


# ===========================================================================
# CONFIGURATION — Edit before each run
# ===========================================================================

# Ground truth object labels (physical objects in the square formation)
GT_OBJECTS = [
    'orange chair',
    'black chair',
    'blue chair',
    'yellow chair',
]

# Keywords that indicate a correct chair detection
# A detected label is TP if it contains any of these keywords
# combined with 'chair' — e.g. 'yellow chair', 'office chair' etc.
# Adjust based on what VLM produces for your environment.
CHAIR_KEYWORDS = [
    'orange chair',
    'black chair',
    'blue chair',
    'yellow chair',
]

# ===========================================================================


def load_object_stack(path):
    with open(path, 'r') as f:
        return json.load(f)


def match_label_to_gt(detected_label, gt_objects):
    """
    Check if a detected label matches any GT object.
    Returns the matched GT label or None.
    Matching is based on exact string match first,
    then partial match (detected label contains GT label).
    """
    detected_lower = detected_label.lower().strip()

    # Exact match
    for gt in gt_objects:
        if detected_lower == gt.lower():
            return gt

    # Partial match — detected label contains GT label
    for gt in gt_objects:
        if gt.lower() in detected_lower:
            return gt

    return None


def evaluate_f1(vlm_stack_path, run_name, output_dir):
    print(f"\n{'='*60}")
    print(f"  F1 + MCS Evaluation — {run_name}")
    print(f"{'='*60}")

    os.makedirs(output_dir, exist_ok=True)

    object_stack = load_object_stack(vlm_stack_path)
    print(f"\n  Loaded {len(object_stack)} objects from: {vlm_stack_path}")
    print(f"  Objects in stack: {list(object_stack.keys())}")
    print(f"\n  GT objects: {GT_OBJECTS}")

    # --- Match detections to GT ---
    tp_labels      = []   # detected labels that match a GT object
    fp_labels      = []   # detected labels that do not match any GT
    matched_gt     = set()  # GT objects that were matched

    print(f"\n  Detection matching:")
    for det_label in object_stack.keys():
        matched = match_label_to_gt(det_label, GT_OBJECTS)
        if matched:
            tp_labels.append(det_label)
            matched_gt.add(matched)
            print(f"    TP: '{det_label}' -> matched GT '{matched}'")
        else:
            fp_labels.append(det_label)
            print(f"    FP: '{det_label}' -> no GT match (noise)")

    # FN = GT objects not matched
    fn_labels = [gt for gt in GT_OBJECTS if gt not in matched_gt]
    for fn in fn_labels:
        print(f"    FN: '{fn}' -> not detected")

    # --- Compute metrics ---
    tp = len(tp_labels)
    fp = len(fp_labels)
    fn = len(fn_labels)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    mcs       = tp / len(GT_OBJECTS) if len(GT_OBJECTS) > 0 else 0.0

    print(f"\n  Results:")
    print(f"  {'Metric':<20} {'Value'}")
    print(f"  {'-'*35}")
    print(f"  {'TP':<20} {tp}")
    print(f"  {'FP':<20} {fp}")
    print(f"  {'FN':<20} {fn}")
    print(f"  {'Precision':<20} {precision:.4f}")
    print(f"  {'Recall':<20} {recall:.4f}")
    print(f"  {'F1-Score':<20} {f1:.4f}")
    print(f"  {'MCS':<20} {mcs:.4f}")

    # --- Save results ---
    results = {
        'run_name':       run_name,
        'gt_objects':     GT_OBJECTS,
        'tp_labels':      tp_labels,
        'fp_labels':      fp_labels,
        'fn_labels':      fn_labels,
        'tp':             tp,
        'fp':             fp,
        'fn':             fn,
        'precision':      round(precision, 4),
        'recall':         round(recall, 4),
        'f1_score':       round(f1, 4),
        'mcs':            round(mcs, 4),
    }

    out_path = os.path.join(output_dir, f'{run_name}_f1_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved: {out_path}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description='F1 + MCS Evaluation for Real-World Semantic Mapping'
    )
    parser.add_argument('--vlm_stack',  required=True,
                        help='Path to vlm_object_stack.json')
    parser.add_argument('--run_name',   required=True,
                        help='Run name for output file')
    parser.add_argument('--output_dir', required=True,
                        help='Directory to save results')
    args = parser.parse_args()

    evaluate_f1(
        vlm_stack_path=args.vlm_stack,
        run_name=args.run_name,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()
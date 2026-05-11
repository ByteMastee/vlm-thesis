#!/usr/bin/env python3
"""
Spatial Position Error (SPE) Evaluation Script
Real-World Pairwise Distance Error for Semantic Mapping Pipelines

Usage:
    python3 evaluate_spe.py \
        --vlm_stack  <path_to_vlm_object_stack.json> \
        --run_name   <run_name> \
        --output_dir <path_to_save_results>

Configuration:
    Edit the LABEL_TO_CORNER and GT_EDGE_DISTANCE variables below
    before running. LABEL_TO_CORNER maps the predicted VLM label
    (as it appears in the vlm_object_stack.json) to a GT corner
    name (A, B, C, D).

Square Formation (edges only):
    A --- B
    |     |
    D --- C
    Edges: AB, BC, CD, DA — all equal to GT_EDGE_DISTANCE metres.

Per-Object SPE:
    SPE_A = sqrt( (d'_AB - GT)^2 + (d'_AD - GT)^2 )
    SPE_B = sqrt( (d'_AB - GT)^2 + (d'_BC - GT)^2 )
    SPE_C = sqrt( (d'_BC - GT)^2 + (d'_CD - GT)^2 )
    SPE_D = sqrt( (d'_CD - GT)^2 + (d'_DA - GT)^2 )

Overall SPE = mean of all computed per-object SPEs.
"""

import os
import json
import argparse
import math
from itertools import combinations


# ===========================================================================
# CONFIGURATION — Edit before each run
# ===========================================================================

# Map predicted VLM label (exact string from vlm_object_stack.json)
# to GT corner name. Use None if an object was not detected.
# Example:
#   'orange chair' -> 'A'
#   'yellow chair' -> 'B'
#   'blue chair'   -> 'C'
#   'black chair'  -> 'D'

LABEL_TO_CORNER = {
    'yellow chair_2': 'A', # <- replace with actual detected label
    #'black chair': 'B', # <- uncomment and fill when detected
    'yellow chair_3':   'C', # <- uncomment and fill when detected
    'yellow chair_1':  'D', # <- uncomment and fill when detected
}

# Ground truth edge distance in metres (all 4 edges of the square)
GT_EDGE_DISTANCE = 3.0

# Square edge pairs (A-B-C-D clockwise)
EDGE_PAIRS = [('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'A')]

# Per-object participating edge pairs
OBJECT_EDGES = {
    'A': [('A', 'B'), ('D', 'A')],
    'B': [('A', 'B'), ('B', 'C')],
    'C': [('B', 'C'), ('C', 'D')],
    'D': [('C', 'D'), ('D', 'A')],
}

# ===========================================================================


def euclidean_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def load_object_stack(path):
    with open(path, 'r') as f:
        return json.load(f)


def build_corner_positions(object_stack, label_to_corner):
    """
    Map GT corner names to predicted (x, y) positions.
    Returns dict: { 'A': (x, y), 'B': (x, y), ... } for detected corners only.
    """
    corner_positions = {}
    for label, corner in label_to_corner.items():
        if label in object_stack:
            x = object_stack[label]['x']
            y = object_stack[label]['y']
            corner_positions[corner] = (x, y)
        else:
            print(f"  [WARN] Label '{label}' (corner {corner}) not found in object stack.")
    return corner_positions


def compute_pairwise_distances(corner_positions):
    """
    Compute predicted distances for all edge pairs where both corners detected.
    Returns dict: { ('A','B'): d_pred, ... }
    """
    pred_distances = {}
    for c1, c2 in EDGE_PAIRS:
        if c1 in corner_positions and c2 in corner_positions:
            x1, y1 = corner_positions[c1]
            x2, y2 = corner_positions[c2]
            pred_distances[(c1, c2)] = euclidean_distance(x1, y1, x2, y2)
    return pred_distances


def compute_per_object_spe(corner_positions, pred_distances, gt_dist):
    """
    Compute per-object SPE using only the edge pairs the object participates in
    and where both endpoints are detected.
    """
    per_object_spe = {}

    for corner, edge_pairs in OBJECT_EDGES.items():
        if corner not in corner_positions:
            continue

        errors_sq = []
        for pair in edge_pairs:
            # Normalize pair order to match EDGE_PAIRS keys
            if pair in pred_distances:
                d_pred = pred_distances[pair]
            elif (pair[1], pair[0]) in pred_distances:
                d_pred = pred_distances[(pair[1], pair[0])]
            else:
                continue  # Partner corner not detected — skip this pair

            error = d_pred - gt_dist
            errors_sq.append(error ** 2)

        if errors_sq:
            spe = math.sqrt(sum(errors_sq))
            per_object_spe[corner] = {
                'spe':            round(spe, 4),
                'num_pairs_used': len(errors_sq),
                'position':       corner_positions[corner]
            }

    return per_object_spe


def compute_overall_spe(per_object_spe):
    if not per_object_spe:
        return None
    spe_values = [v['spe'] for v in per_object_spe.values()]
    return round(sum(spe_values) / len(spe_values), 4)


def evaluate(vlm_stack_path, run_name, output_dir):
    print(f"\n{'='*60}")
    print(f"  SPE Evaluation — {run_name}")
    print(f"{'='*60}")

    os.makedirs(output_dir, exist_ok=True)

    object_stack = load_object_stack(vlm_stack_path)
    print(f"\n  Loaded {len(object_stack)} objects from: {vlm_stack_path}")
    print(f"  Objects in stack: {list(object_stack.keys())}")

    print(f"\n  Label-to-corner mapping:")
    for label, corner in LABEL_TO_CORNER.items():
        status = 'FOUND' if label in object_stack else 'MISSING'
        print(f"    {corner} <- '{label}' [{status}]")

    corner_positions = build_corner_positions(object_stack, LABEL_TO_CORNER)
    print(f"\n  Detected corners: {list(corner_positions.keys())}")

    if len(corner_positions) < 2:
        print("\n  [ERROR] Need at least 2 detected corners to compute any edge distance.")
        return

    pred_distances = compute_pairwise_distances(corner_positions)

    print(f"\n  Predicted vs GT edge distances (GT = {GT_EDGE_DISTANCE}m):")
    print(f"  {'Edge':<10} {'Predicted (m)':<18} {'GT (m)':<10} {'Error (m)':<12}")
    print(f"  {'-'*50}")
    for (c1, c2), d_pred in pred_distances.items():
        error = d_pred - GT_EDGE_DISTANCE
        print(f"  {c1}-{c2:<8} {d_pred:<18.4f} {GT_EDGE_DISTANCE:<10.4f} {error:<12.4f}")

    per_object_spe = compute_per_object_spe(corner_positions, pred_distances, GT_EDGE_DISTANCE)
    overall_spe    = compute_overall_spe(per_object_spe)

    print(f"\n  Per-Object SPE:")
    print(f"  {'Corner':<10} {'SPE (m)':<12} {'Pairs Used':<12} {'Position'}")
    print(f"  {'-'*55}")
    for corner, data in per_object_spe.items():
        pos = data['position']
        print(f"  {corner:<10} {data['spe']:<12.4f} {data['num_pairs_used']:<12} ({pos[0]:.4f}, {pos[1]:.4f})")

    print(f"\n  Overall SPE (mean): {overall_spe} m")

    # Save results
    results = {
        'run_name':          run_name,
        'gt_edge_distance':  GT_EDGE_DISTANCE,
        'label_to_corner':   LABEL_TO_CORNER,
        'detected_corners':  {k: list(v) for k, v in corner_positions.items()},
        'predicted_edges':   {f"{c1}-{c2}": round(d, 4) for (c1, c2), d in pred_distances.items()},
        'per_object_spe':    per_object_spe,
        'overall_spe':       overall_spe,
    }

    out_path = os.path.join(output_dir, f'{run_name}_spe_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved: {out_path}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='SPE Evaluation for Semantic Mapping')
    parser.add_argument('--vlm_stack',  required=True, help='Path to vlm_object_stack.json')
    parser.add_argument('--run_name',   required=True, help='Run name for output file')
    parser.add_argument('--output_dir', required=True, help='Directory to save results')
    args = parser.parse_args()

    evaluate(
        vlm_stack_path=args.vlm_stack,
        run_name=args.run_name,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()
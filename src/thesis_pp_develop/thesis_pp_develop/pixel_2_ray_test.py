#!/usr/bin/env python3

import numpy as np

# Camera intrinsics from camera_info
fx = 28.00600204423685
fy = 28.00600204423685
cx_0 = 320.5
cy_0 = 240.5

# Rotation matrix: optical_frame -> base_link (from extract_static_tf.py)
R_optical_to_base = np.array([
    [-2.55002079e-02, -9.99674817e-01,  2.15810911e-06],
    [-5.87339488e-01,  1.49804041e-02, -8.09202023e-01],
    [ 8.08938852e-01, -2.06360874e-02, -5.87530498e-01]
])


def pixel_to_ray_camera(cx, cy):
    x = (cx - cx_0) / fx
    y = (cy - cy_0) / fy
    z = 1.0
    ray = np.array([x, y, z])
    ray = ray / np.linalg.norm(ray)
    return ray


def ray_camera_to_base(ray_camera):
    ray_base = R_optical_to_base @ ray_camera
    ray_base = ray_base / np.linalg.norm(ray_base)
    return ray_base


# Test with sample centroids from detections.json
test_centroids = [
    (471, 197),
    (320, 240),  # image center — should point straight ahead
    (100, 300),
    (550, 150),
]

for (cx, cy) in test_centroids:
    ray_cam = pixel_to_ray_camera(cx, cy)
    ray_base = ray_camera_to_base(ray_cam)
    print(f'Pixel ({cx:3d}, {cy:3d}) -> '
          f'ray_cam: [{ray_cam[0]:+.4f}, {ray_cam[1]:+.4f}, {ray_cam[2]:+.4f}] -> '
          f'ray_base: [{ray_base[0]:+.4f}, {ray_base[1]:+.4f}, {ray_base[2]:+.4f}]')
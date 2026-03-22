#!/usr/bin/env python3

import numpy as np
import json
import cv2
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import rosbag2_py
from ultralytics import YOLO
from sklearn.cluster import DBSCAN

BAG_PATH = '/root/UVC_ws/vf_robot_model_ros2/thesis_fisheye_bag3'
IMAGE_TOPIC = '/fisheye/front/fisheye_front/image_raw'
ODOM_TOPIC = '/odom'
FRAME_SKIP = 12
CONFIDENCE = 0.45
MODEL_PATH = '/root/yolo26m.pt'
MIN_ANGLE_DEG = 5.0
OUTPUT_PATH = '/root/UVC_ws/vf_robot_model_ros2/step7_detections/object_stack_sample.json'

# Camera intrinsics
FX = 28.00600204423685
FY = 28.00600204423685
CX_0 = 320.5
CY_0 = 240.5

# Rotation matrix: optical_frame -> base_link
R_optical_to_base = np.array([
    [-2.55002079e-02, -9.99674817e-01,  2.15810911e-06],
    [-5.87339488e-01,  1.49804041e-02, -8.09202023e-01],
    [ 8.08938852e-01, -2.06360874e-02, -5.87530498e-01]
])

# Camera offset from base_link
T_CAM_OFFSET = np.array([7.00000000e-02, 0.00000000e+00, 1.84500000e+00])


def pixel_to_ray_odom(cx, cy, robot_x, robot_y, yaw):
    x = (cx - CX_0) / FX
    y = (cy - CY_0) / FY
    z = 1.0
    ray_cam = np.array([x, y, z])
    ray_cam = ray_cam / np.linalg.norm(ray_cam)

    ray_base = R_optical_to_base @ ray_cam
    ray_base = ray_base / np.linalg.norm(ray_base)

    c, s = np.cos(yaw), np.sin(yaw)
    R_yaw = np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ])
    ray_odom = R_yaw @ ray_base
    ray_odom = ray_odom / np.linalg.norm(ray_odom)

    R2d = np.array([[c, -s], [s, c]])
    cam_offset_rotated = R2d @ T_CAM_OFFSET[:2]
    origin = np.array([
        robot_x + cam_offset_rotated[0],
        robot_y + cam_offset_rotated[1],
        T_CAM_OFFSET[2]
    ])

    return origin, ray_odom


def angle_between_rays(d1, d2):
    cos_angle = np.clip(np.dot(d1, d2), -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


def closest_approach_midpoint(o1, d1, o2, d2):
    # Find midpoint of closest approach between two rays
    # o1 + t1*d1 and o2 + t2*d2
    w0 = o1 - o2
    a = np.dot(d1, d1)
    b = np.dot(d1, d2)
    c = np.dot(d2, d2)
    d = np.dot(d1, w0)
    e = np.dot(d2, w0)

    denom = a * c - b * b
    if abs(denom) < 1e-6:
        return None

    t1 = (b * e - c * d) / denom
    t2 = (a * e - b * d) / denom

    if t1 < 0 or t2 < 0:
        return None

    p1 = o1 + t1 * d1
    p2 = o2 + t2 * d2
    midpoint = (p1 + p2) / 2.0

    return midpoint


# Read bag
storage_options = rosbag2_py.StorageOptions(uri=BAG_PATH, storage_id='sqlite3')
converter_options = rosbag2_py.ConverterOptions(
    input_serialization_format='cdr',
    output_serialization_format='cdr'
)

reader = rosbag2_py.SequentialReader()
reader.open(storage_options, converter_options)

topic_types = reader.get_all_topics_and_types()
type_map = {t.name: t.type for t in topic_types}

storage_filter = rosbag2_py.StorageFilter(topics=[IMAGE_TOPIC, ODOM_TOPIC])
reader.set_filter(storage_filter)

image_msg_type = get_message(type_map[IMAGE_TOPIC])
odom_msg_type = get_message(type_map[ODOM_TOPIC])

model = YOLO(MODEL_PATH)

# Ray stack: {label: [(origin, ray_direction), ...]}
ray_stack = {}

# Candidate points: {label: [(x, y), ...]}
candidate_stack = {}

frame_count = 0
latest_odom = None
total_pairs_triangulated = 0
total_pairs_skipped = 0

while reader.has_next():
    topic, data, timestamp = reader.read_next()

    if topic == ODOM_TOPIC:
        latest_odom = deserialize_message(data, odom_msg_type)
        continue

    if topic != IMAGE_TOPIC:
        continue

    if frame_count % FRAME_SKIP == 0 and latest_odom is not None:
        msg = deserialize_message(data, image_msg_type)
        img_array = np.frombuffer(bytes(msg.data), dtype=np.uint8)

        if msg.encoding == 'rgb8':
            img = img_array.reshape((msg.height, msg.width, 3))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            frame_count += 1
            continue

        results = model(img, conf=CONFIDENCE, verbose=False)

        rx = latest_odom.pose.pose.position.x
        ry = latest_odom.pose.pose.position.y
        q = latest_odom.pose.pose.orientation
        yaw = np.arctan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y**2 + q.z**2))

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                origin, ray = pixel_to_ray_odom(cx, cy, rx, ry, yaw)

                # Add to ray stack
                if label not in ray_stack:
                    ray_stack[label] = []
                if label not in candidate_stack:
                    candidate_stack[label] = []

                # Triangulate with all existing rays of same label
                for prev_origin, prev_ray in ray_stack[label]:
                    angle = angle_between_rays(ray, prev_ray)

                    if angle < MIN_ANGLE_DEG:
                        total_pairs_skipped += 1
                        print(f'  Skipped pair [{label}] angle={angle:.2f}deg < {MIN_ANGLE_DEG}deg')
                        continue

                    midpoint = closest_approach_midpoint(prev_origin, prev_ray, origin, ray)

                    if midpoint is None:
                        continue

                    candidate_stack[label].append((midpoint[0], midpoint[1]))
                    total_pairs_triangulated += 1
                    print(f'  Triangulated [{label}] angle={angle:.2f}deg -> '
                          f'candidate: ({midpoint[0]:.3f}, {midpoint[1]:.3f})')

                ray_stack[label].append((origin, ray))

    frame_count += 1

print(f'\nTotal pairs triangulated: {total_pairs_triangulated}')
print(f'Total pairs skipped (angle < {MIN_ANGLE_DEG}deg): {total_pairs_skipped}')

# DBSCAN parameters
DBSCAN_EPS = 1.0        # max distance between points in same cluster (meters)
DBSCAN_MIN_SAMPLES = 3  # minimum points to form a cluster

object_stack = {}

print('\n--- Object Stack (after DBSCAN clustering) ---')
for label, candidates in candidate_stack.items():
    if len(candidates) == 0:
        continue

    pts = np.array(candidates)

    if len(pts) < DBSCAN_MIN_SAMPLES:
        print(f'{label}: not enough candidates ({len(pts)}) for clustering — skipped')
        continue

    db = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit(pts)
    labels_db = db.labels_

    unique_clusters = set(labels_db)
    unique_clusters.discard(-1)  # remove noise label

    if len(unique_clusters) == 0:
        print(f'{label}: all points marked as noise — skipped')
        continue

    for cluster_id in sorted(unique_clusters):
        cluster_pts = pts[labels_db == cluster_id]
        final_x = float(np.median(cluster_pts[:, 0]))
        final_y = float(np.median(cluster_pts[:, 1]))

        if len(unique_clusters) == 1:
            instance_label = label
        else:
            instance_label = f'{label}_{cluster_id + 1}'

        object_stack[instance_label] = {
            'x': round(final_x, 4),
            'y': round(final_y, 4),
            'num_candidates': len(cluster_pts)
        }

        print(f'{instance_label}: ({final_x:.4f}, {final_y:.4f}) '
              f'from {len(cluster_pts)} candidates')

    noise_count = np.sum(labels_db == -1)
    if noise_count > 0:
        print(f'  [{label}] noise points discarded: {noise_count}')

# Save
import os
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
with open(OUTPUT_PATH, 'w') as f:
    json.dump(object_stack, f, indent=2)

print(f'\nObject stack saved to: {OUTPUT_PATH}')


#Only if ray stack is needed, use the folowing code:

# Save ray stack
ray_stack_serializable = {}
for label, rays in ray_stack.items():
    ray_stack_serializable[label] = [
        {
            'origin': [round(float(o[0]), 4), round(float(o[1]), 4), round(float(o[2]), 4)],
            'direction': [round(float(d[0]), 4), round(float(d[1]), 4), round(float(d[2]), 4)]
        }
        for o, d in rays
    ]

ray_stack_path = os.path.join(os.path.dirname(OUTPUT_PATH), 'ray_stack.json')
with open(ray_stack_path, 'w') as f:
    json.dump(ray_stack_serializable, f, indent=2)

print(f'Ray stack saved to: {ray_stack_path}')
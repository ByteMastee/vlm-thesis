#!/usr/bin/env python3

import os
import time
import json
import cv2
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN

import rclpy
from rclpy.node import Node
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import rosbag2_py
from ultralytics import YOLO


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

# SIFT parameters
MIN_MATCH_COUNT = 4
LOWE_RATIO = 0.75

# Triangulation parameters
MIN_ANGLE_DEG = 5.0

# DBSCAN parameters
DBSCAN_EPS = 1.0
DBSCAN_MIN_SAMPLES = 3

# Filter parameters
MIN_CLUSTER_CANDIDATES = 4
MAP_BOUNDS = 6.0  # metres from origin

# Ground truth from Gazebo world file
GROUND_TRUTH = {
    'chair_1': (-3.0, 2.0),
    'chair_2': (-3.5, -2.5),
    'couch':   ( 3.5, 0.0),
    'table':   ( 2.0, 2.5)
}


def get_camera_origin_in_odom(robot_x, robot_y, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    R2d = np.array([[c, -s], [s, c]])
    cam_offset_rotated = R2d @ T_CAM_OFFSET[:2]
    origin = np.array([
        robot_x + cam_offset_rotated[0],
        robot_y + cam_offset_rotated[1],
        T_CAM_OFFSET[2]
    ])
    return origin


def pixel_to_ray_odom(px, py, robot_x, robot_y, yaw):
    x = (px - CX_0) / FX
    y = (py - CY_0) / FY
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

    origin = get_camera_origin_in_odom(robot_x, robot_y, yaw)

    return origin, ray_odom


def angle_between_rays(d1, d2):
    cos_angle = np.clip(np.dot(d1, d2), -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


def closest_approach_midpoint(o1, d1, o2, d2):
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


def extract_sift_keypoints(img, bbox):
    x1, y1, x2, y2 = bbox
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img.shape[1], x2)
    y2 = min(img.shape[0], y2)

    if x2 <= x1 or y2 <= y1:
        return None, None, None

    crop = img[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # Upscale small crops so SIFT has enough pixels to detect features
    orig_h, orig_w = gray.shape
    scale_x, scale_y = 1.0, 1.0
    if orig_h < 80 or orig_w < 80:
        scale_x = max(80 / orig_w, 1.0)
        scale_y = max(80 / orig_h, 1.0)
        gray = cv2.resize(
            gray,
            (int(orig_w * scale_x), int(orig_h * scale_y)),
            interpolation=cv2.INTER_CUBIC
        )

    # CLAHE contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    gray = clahe.apply(gray)

    sift = cv2.SIFT_create()
    kps, descs = sift.detectAndCompute(gray, None)

    if kps is None or descs is None or len(kps) == 0:
        print(f'SIFT: 0 keypoints in crop size {crop.shape}')
        return None, None, None

    print(f'SIFT: {len(kps)} keypoints in crop size {crop.shape}')

    # Convert keypoint coords back to full image coords (undo upscale + bbox offset)
    kps_full = []
    for kp in kps:
        full_x = (kp.pt[0] / scale_x) + x1
        full_y = (kp.pt[1] / scale_y) + y1
        kps_full.append((full_x, full_y))

    return kps_full, descs, crop


def match_sift_features(descs1, descs2):
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = bf.knnMatch(descs1, descs2, k=2)

    good = []
    for m, n in matches:
        if m.distance < LOWE_RATIO * n.distance:
            good.append(m)

    return good


def cluster_candidates(candidates, label):
    object_entries = {}

    if len(candidates) == 0:
        return object_entries

    pts = np.array(candidates)

    if len(pts) < DBSCAN_MIN_SAMPLES:
        final_x = float(np.median(pts[:, 0]))
        final_y = float(np.median(pts[:, 1]))
        object_entries[label] = {
            'x': round(final_x, 4),
            'y': round(final_y, 4),
            'num_candidates': len(pts)
        }
        return object_entries

    db = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit(pts)
    labels_db = db.labels_

    unique_clusters = set(labels_db)
    unique_clusters.discard(-1)

    if len(unique_clusters) == 0:
        return object_entries

    for cluster_id in sorted(unique_clusters):
        cluster_pts = pts[labels_db == cluster_id]
        final_x = float(np.median(cluster_pts[:, 0]))
        final_y = float(np.median(cluster_pts[:, 1]))

        if len(unique_clusters) == 1:
            instance_label = label
        else:
            instance_label = f'{label}_{cluster_id + 1}'

        object_entries[instance_label] = {
            'x': round(final_x, 4),
            'y': round(final_y, 4),
            'num_candidates': len(cluster_pts)
        }

    return object_entries

def filter_object_stack(object_stack):
    filtered = {}
    for label, data in object_stack.items():
        # Filter 1: minimum candidates
        if data['num_candidates'] < MIN_CLUSTER_CANDIDATES:
            print(f'FILTER: {label} removed — only {data["num_candidates"]} candidates')
            continue
        # Filter 2: spatial bounds
        if abs(data['x']) > MAP_BOUNDS or abs(data['y']) > MAP_BOUNDS:
            print(f'FILTER: {label} removed — out of bounds ({data["x"]}, {data["y"]})')
            continue
        filtered[label] = data
    return filtered

def save_map_plot(object_stack, output_dir, robot_x=None, robot_y=None):
    plt.figure(figsize=(10, 10))

    if robot_x and robot_y:
        plt.plot(robot_x, robot_y, 'b-', linewidth=1.0, alpha=0.5)
        plt.plot(robot_x[0], robot_y[0], 'go', markersize=8)
        plt.plot(robot_x[-1], robot_y[-1], 'rs', markersize=8)

    for label, (gx, gy) in GROUND_TRUTH.items():
        plt.plot(gx, gy, 'g^', markersize=12)
        plt.annotate(f'GT: {label}\n({gx},{gy})', (gx, gy),
                     textcoords='offset points', xytext=(8, 8),
                     fontsize=9, color='green')

    colors = ['red', 'orange', 'purple', 'cyan', 'magenta']
    for i, (label, data) in enumerate(object_stack.items()):
        ox = data['x']
        oy = data['y']
        color = colors[i % len(colors)]
        plt.plot(ox, oy, '*', markersize=15, color=color)
        plt.annotate(f'Det: {label}\n({ox:.2f},{oy:.2f})', (ox, oy),
                     textcoords='offset points', xytext=(8, -18),
                     fontsize=9, color=color)

        best_dist = float('inf')
        best_gx, best_gy = None, None
        for gt_label, (gx, gy) in GROUND_TRUTH.items():
            dist = np.sqrt((ox - gx)**2 + (oy - gy)**2)
            if dist < best_dist:
                best_dist = dist
                best_gx, best_gy = gx, gy

        if best_gx is not None:
            plt.plot([ox, best_gx], [oy, best_gy], '--', color=color, linewidth=1.0)
            plt.text((ox + best_gx) / 2, (oy + best_gy) / 2,
                     f'{best_dist:.2f}m', fontsize=8, color=color)

    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('Semantic Map — Detected vs Ground Truth (SIFT)')
    plt.legend(handles=[
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='green',
                   markersize=10, label='Ground Truth'),
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='red',
                   markersize=10, label='Detected'),
        plt.Line2D([0], [0], color='blue', linewidth=1.0, label='Robot path'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green',
                   markersize=8, label='Start'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='red',
                   markersize=8, label='End')
    ])
    plt.grid(True)
    plt.axis('equal')

    plot_path = os.path.join(output_dir, 'map_plot5.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()

    return plot_path


class FrameProcessor(Node):
    def __init__(self):
        super().__init__('frame_processing_node')

        self.declare_parameter('bag_path', '/root/UVC_ws/vf_robot_model_ros2/thesis_fisheye_bag3')
        self.declare_parameter('image_topic', '/fisheye/front/fisheye_front/image_raw')
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('frame_skip', 12)
        self.declare_parameter('confidence', 0.45)
        self.declare_parameter('model_path', '/root/yolo26m.pt')
        self.declare_parameter('output_dir', '/root/UVC_ws/vf_robot_model_ros2/pp_tunning')

        bag_path      = self.get_parameter('bag_path').value
        image_topic   = self.get_parameter('image_topic').value
        odom_topic    = self.get_parameter('odom_topic').value
        frame_skip    = self.get_parameter('frame_skip').value
        confidence    = self.get_parameter('confidence').value
        model_path    = self.get_parameter('model_path').value
        output_dir    = self.get_parameter('output_dir').value

        os.makedirs(output_dir, exist_ok=True)

        self.get_logger().info(f'Reading bag: {bag_path}')
        self.get_logger().info(f'Frame skip: {frame_skip}')
        self.get_logger().info(f'Loading YOLO model: {model_path}')

        model = YOLO(model_path)

        storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id='sqlite3')
        converter_options = rosbag2_py.ConverterOptions(
            input_serialization_format='cdr',
            output_serialization_format='cdr'
        )

        reader = rosbag2_py.SequentialReader()
        reader.open(storage_options, converter_options)

        topic_types = reader.get_all_topics_and_types()
        type_map = {t.name: t.type for t in topic_types}

        storage_filter = rosbag2_py.StorageFilter(topics=[image_topic, odom_topic])
        reader.set_filter(storage_filter)

        image_msg_type = get_message(type_map[image_topic])
        odom_msg_type  = get_message(type_map[odom_topic])

        frame_count   = 0
        process_count = 0
        latest_odom   = None

        # Per-label storage: list of (kps_full, descs, rx, ry, yaw)
        feature_stack   = {}
        candidate_stack = {}

        total_pairs_triangulated = 0
        total_pairs_skipped      = 0
        robot_x_list, robot_y_list = [], []

        loop_start_time = time.time()

        while reader.has_next():
            topic, data, timestamp = reader.read_next()

            if topic == odom_topic:
                latest_odom = deserialize_message(data, odom_msg_type)
                robot_x_list.append(latest_odom.pose.pose.position.x)
                robot_y_list.append(latest_odom.pose.pose.position.y)
                continue

            if topic != image_topic:
                continue

            if frame_count % frame_skip == 0 and latest_odom is not None:
                msg = deserialize_message(data, image_msg_type)
                img_array = np.frombuffer(bytes(msg.data), dtype=np.uint8)

                if msg.encoding == 'rgb8':
                    img = img_array.reshape((msg.height, msg.width, 3))
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                elif msg.encoding == 'bgr8':
                    img = img_array.reshape((msg.height, msg.width, 3))
                elif msg.encoding == 'mono8':
                    img = img_array.reshape((msg.height, msg.width))
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                else:
                    self.get_logger().warn(f'Unsupported encoding: {msg.encoding}')
                    frame_count += 1
                    continue

                results = model(img, conf=confidence, verbose=False)

                rx = latest_odom.pose.pose.position.x
                ry = latest_odom.pose.pose.position.y
                q  = latest_odom.pose.pose.orientation
                yaw = np.arctan2(
                    2 * (q.w * q.z + q.x * q.y),
                    1 - 2 * (q.y**2 + q.z**2)
                )

                for result in results:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cls_id = int(box.cls[0])
                        label  = model.names[cls_id]

                        # Extract SIFT keypoints inside YOLO bounding box only
                        kps_full, descs, _ = extract_sift_keypoints(
                            img, (x1, y1, x2, y2)
                        )

                        if kps_full is None:
                            continue

                        if label not in feature_stack:
                            feature_stack[label]   = []
                            candidate_stack[label] = []

                        # Match current frame keypoints against all previous
                        # frames for the same label
                        for prev_kps, prev_descs, prev_rx, prev_ry, prev_yaw \
                                in feature_stack[label]:

                            if len(descs) < 2 or len(prev_descs) < 2:
                                continue

                            good_matches = match_sift_features(prev_descs, descs)

                            if len(good_matches) < MIN_MATCH_COUNT:
                                total_pairs_skipped += 1
                                continue

                            # Triangulate each matched keypoint pair
                            for m in good_matches:
                                px1, py1 = prev_kps[m.queryIdx]
                                px2, py2 = kps_full[m.trainIdx]

                                o1, d1 = pixel_to_ray_odom(
                                    px1, py1, prev_rx, prev_ry, prev_yaw
                                )
                                o2, d2 = pixel_to_ray_odom(
                                    px2, py2, rx, ry, yaw
                                )

                                angle = angle_between_rays(d1, d2)
                                if angle < MIN_ANGLE_DEG:
                                    continue

                                midpoint = closest_approach_midpoint(
                                    o1, d1, o2, d2
                                )
                                if midpoint is None:
                                    continue

                                candidate_stack[label].append(
                                    (midpoint[0], midpoint[1])
                                )
                                total_pairs_triangulated += 1

                        feature_stack[label].append(
                            (kps_full, descs, rx, ry, yaw)
                        )

                process_count += 1
                self.get_logger().info(
                    f'Frame {frame_count} processed | '
                    f'robot: ({rx:.3f},{ry:.3f})'
                )

            frame_count += 1

        loop_elapsed = time.time() - loop_start_time

        self.get_logger().info(f'Total frames in bag: {frame_count}')
        self.get_logger().info(f'Frames processed: {process_count}')
        self.get_logger().info(f'Pairs triangulated: {total_pairs_triangulated}')
        self.get_logger().info(f'Pairs skipped: {total_pairs_skipped}')
        self.get_logger().info(
            f'Processing loop time: {loop_elapsed:.3f}s '
            f'({loop_elapsed/60:.2f} min)'
        )

        # DBSCAN clustering -> final object positions
        object_stack = {}
        for label, candidates in candidate_stack.items():
            entries = cluster_candidates(candidates, label)
            object_stack.update(entries)

        object_stack = filter_object_stack(object_stack)

        for label, data in object_stack.items():
            self.get_logger().info(
                f'Object: {label} -> ({data["x"]}, {data["y"]}) '
                f'from {data["num_candidates"]} candidates'
            )

        # Save object stack
        json_path = os.path.join(output_dir, 'object_stack5.json')
        with open(json_path, 'w') as f:
            json.dump(object_stack, f, indent=2)
        self.get_logger().info(f'Object stack saved to: {json_path}')

        # Save robot path
        robot_path_data = {'x': robot_x_list, 'y': robot_y_list}
        robot_path_json = os.path.join(output_dir, 'robot_path5.json')
        with open(robot_path_json, 'w') as f:
            json.dump(robot_path_data, f)
        self.get_logger().info(f'Robot path saved to: {robot_path_json}')

        # Save map plot
        plot_path = save_map_plot(
            object_stack, output_dir, robot_x_list, robot_y_list
        )
        self.get_logger().info(f'Map plot saved to: {plot_path}')


def main(args=None):
    rclpy.init(args=args)
    node = FrameProcessor()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
#!/usr/bin/env python3

import os
import time
import json
import cv2
import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from PIL import Image
from sklearn.cluster import HDBSCAN
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

import rclpy
from rclpy.node import Node
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import rosbag2_py
from ultralytics import YOLO


# Camera intrinsics
FX = 554.26
FY = 554.26
CX_0 = 320.5
CY_0 = 240.5

# Rotation matrix: optical_frame -> base_link
R_optical_to_base = np.array([
    [-2.55002079e-02, -9.99674817e-01,  2.15810911e-06],
    [-5.87339488e-01,  1.49804041e-02, -8.09202023e-01],
    [ 8.08938852e-01, -2.06360874e-02, -5.87530498e-01]
])

# Camera translation: optical_frame -> base_link
T_CAM_OFFSET = np.array([1.78103284e-03, 1.53409150e+00, 1.02736805e+00])

# Depth Anything V2 model path
DEPTH_MODEL_PATH = '/root/depth_anything_v2_metric_indoor_small'

# Minimum bounding box area to use depth (pixels squared)
MIN_BBOX_AREA = 1800

# HDBSCAN parameters
HDBSCAN_MIN_CLUSTER_SIZE = 3
HDBSCAN_MIN_SAMPLES = 2

# Ground truth from Gazebo world file
GROUND_TRUTH = {
    'chair_1': (-3.0, 2.0),
    'chair_2': (-3.5, -2.5),
    'couch':   ( 3.5, 0.0),
    'table':   ( 2.0, 2.5)
}


def depth_to_3d_odom(cx, cy, depth_m, robot_x, robot_y, yaw):
    # Standard pinhole unprojection
    x_cam = (cx - CX_0) * depth_m / FX
    y_cam = (cy - CY_0) * depth_m / FY
    z_cam = depth_m
    point_cam = np.array([x_cam, y_cam, z_cam])

    # camera optical frame -> base_link
    point_base = R_optical_to_base @ point_cam + T_CAM_OFFSET

    # base_link -> odom
    c, s = np.cos(yaw), np.sin(yaw)
    R_yaw = np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ])
    point_odom = R_yaw @ point_base
    point_odom[0] += robot_x
    point_odom[1] += robot_y

    return point_odom


def cluster_candidates(candidates, label):
    object_entries = {}

    if len(candidates) == 0:
        return object_entries

    pts = np.array(candidates)

    if len(pts) < HDBSCAN_MIN_CLUSTER_SIZE:
        final_x = float(np.median(pts[:, 0]))
        final_y = float(np.median(pts[:, 1]))
        object_entries[label] = {
            'x': round(final_x, 4),
            'y': round(final_y, 4),
            'num_candidates': len(pts)
        }
        return object_entries

    hdb = HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=HDBSCAN_MIN_SAMPLES
    ).fit(pts)
    labels_db = hdb.labels_

    unique_clusters = set(labels_db)
    unique_clusters.discard(-1)

    if len(unique_clusters) == 0:
        # All points classified as noise — use median of all points
        final_x = float(np.median(pts[:, 0]))
        final_y = float(np.median(pts[:, 1]))
        object_entries[label] = {
            'x': round(final_x, 4),
            'y': round(final_y, 4),
            'num_candidates': len(pts)
        }
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
    plt.title('Semantic Map — Detected vs Ground Truth (Depth Anything V2 + HDBSCAN)')
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

    plot_path = os.path.join(output_dir, 'map_plot_depth4.png')
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
        self.declare_parameter('output_dir', '/root/UVC_ws/vf_robot_model_ros2/pp_tunning2')

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
        self.get_logger().info(f'Loading Depth Anything V2 from: {DEPTH_MODEL_PATH}')

        # Load YOLO
        yolo_model = YOLO(model_path)

        # Load Depth Anything V2
        depth_processor = AutoImageProcessor.from_pretrained(
            DEPTH_MODEL_PATH, use_fast=False
        )
        depth_model = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL_PATH)
        depth_model.eval()
        self.get_logger().info('Depth Anything V2 loaded successfully')

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

        frame_count      = 0
        process_count    = 0
        depth_used_count = 0
        bbox_skip_count  = 0
        latest_odom      = None

        candidate_stack = {}

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

                yolo_results = yolo_model(img, conf=confidence, verbose=False)

                rx  = latest_odom.pose.pose.position.x
                ry  = latest_odom.pose.pose.position.y
                q   = latest_odom.pose.pose.orientation
                yaw = np.arctan2(
                    2 * (q.w * q.z + q.x * q.y),
                    1 - 2 * (q.y**2 + q.z**2)
                )

                # Check if any detection passes bbox area filter
                valid_detections = []
                for result in yolo_results:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        bbox_area = (x2 - x1) * (y2 - y1)
                        if bbox_area >= MIN_BBOX_AREA:
                            cls_id = int(box.cls[0])
                            label  = yolo_model.names[cls_id]
                            cx = (x1 + x2) // 2
                            cy = (y1 + y2) // 2
                            valid_detections.append((label, cx, cy, bbox_area))
                        else:
                            bbox_skip_count += 1

                if len(valid_detections) > 0:
                    # Run Depth Anything V2 once per frame
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(img_rgb)
                    inputs = depth_processor(images=pil_img, return_tensors='pt')

                    with torch.no_grad():
                        outputs = depth_model(**inputs)
                        predicted_depth = outputs.predicted_depth

                    # Interpolate depth to original image size
                    depth_map = torch.nn.functional.interpolate(
                        predicted_depth.unsqueeze(1),
                        size=(msg.height, msg.width),
                        mode='bicubic',
                        align_corners=False
                    ).squeeze().numpy()

                    # Process each valid detection
                    for label, cx, cy, bbox_area in valid_detections:
                        depth_m = float(depth_map[cy, cx])

                        # Skip invalid depth values
                        if depth_m <= 0.1 or depth_m > 3.0:
                            continue

                        # Convert to 3D point in odom frame
                        point_odom = depth_to_3d_odom(cx, cy, depth_m, rx, ry, yaw)

                        if label not in candidate_stack:
                            candidate_stack[label] = []

                        candidate_stack[label].append(
                            (point_odom[0], point_odom[1])
                        )
                        depth_used_count += 1

                        self.get_logger().info(
                            f'Frame {frame_count} | {label} | '
                            f'bbox_area: {bbox_area} | '
                            f'depth: {depth_m:.3f}m | '
                            f'3D: ({point_odom[0]:.3f}, {point_odom[1]:.3f})'
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
        self.get_logger().info(f'Depth points used: {depth_used_count}')
        self.get_logger().info(f'Detections skipped (small bbox): {bbox_skip_count}')
        self.get_logger().info(
            f'Processing loop time: {loop_elapsed:.3f}s '
            f'({loop_elapsed/60:.2f} min)'
        )

        # HDBSCAN clustering -> object stack
        object_stack = {}
        for label, candidates in candidate_stack.items():
            entries = cluster_candidates(candidates, label)
            object_stack.update(entries)

        for label, data in object_stack.items():
            self.get_logger().info(
                f'Object: {label} -> ({data["x"]}, {data["y"]}) '
                f'from {data["num_candidates"]} candidates'
            )

        # Save object stack
        json_path = os.path.join(output_dir, 'object_stack_depth4.json')
        with open(json_path, 'w') as f:
            json.dump(object_stack, f, indent=2)
        self.get_logger().info(f'Object stack saved to: {json_path}')

        # Save robot path
        robot_path_data = {'x': robot_x_list, 'y': robot_y_list}
        robot_path_json = os.path.join(output_dir, 'robot_pathDepth.json')
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
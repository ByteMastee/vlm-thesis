#!/usr/bin/env python3

import os
import json
import cv2
import numpy as np
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

# Triangulation parameters
MIN_ANGLE_DEG = 5.0

# DBSCAN parameters
DBSCAN_EPS = 1.0
DBSCAN_MIN_SAMPLES = 3


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


class FrameProcessor(Node):
    def __init__(self):
        super().__init__('frame_processing_node')

        self.declare_parameter('bag_path', '/root/UVC_ws/vf_robot_model_ros2/thesis_fisheye_bag3')
        self.declare_parameter('image_topic', '/fisheye/front/fisheye_front/image_raw')
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('frame_skip', 12)
        self.declare_parameter('confidence', 0.45)
        self.declare_parameter('model_path', '/root/yolo26m.pt')
        self.declare_parameter('output_dir', '/root/UVC_ws/vf_robot_model_ros2/pp_output')

        bag_path = self.get_parameter('bag_path').value
        image_topic = self.get_parameter('image_topic').value
        odom_topic = self.get_parameter('odom_topic').value
        frame_skip = self.get_parameter('frame_skip').value
        confidence = self.get_parameter('confidence').value
        model_path = self.get_parameter('model_path').value
        output_dir = self.get_parameter('output_dir').value

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
        odom_msg_type = get_message(type_map[odom_topic])

        frame_count = 0
        process_count = 0
        latest_odom = None

        # Ray stack: {label: [(origin, ray), ...]}
        ray_stack = {}

        # Candidate stack: {label: [(x, y), ...]}
        candidate_stack = {}

        total_pairs_triangulated = 0
        total_pairs_skipped = 0

        while reader.has_next():
            topic, data, timestamp = reader.read_next()

            if topic == odom_topic:
                latest_odom = deserialize_message(data, odom_msg_type)
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
                q = latest_odom.pose.pose.orientation
                yaw = np.arctan2(
                    2*(q.w*q.z + q.x*q.y),
                    1 - 2*(q.y**2 + q.z**2)
                )

                for result in results:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cls_id = int(box.cls[0])
                        label = model.names[cls_id]
                        cx = (x1 + x2) // 2
                        cy = (y1 + y2) // 2

                        origin, ray = pixel_to_ray_odom(cx, cy, rx, ry, yaw)

                        if label not in ray_stack:
                            ray_stack[label] = []
                        if label not in candidate_stack:
                            candidate_stack[label] = []

                        for prev_origin, prev_ray in ray_stack[label]:
                            angle = angle_between_rays(ray, prev_ray)

                            if angle < MIN_ANGLE_DEG:
                                total_pairs_skipped += 1
                                continue

                            midpoint = closest_approach_midpoint(
                                prev_origin, prev_ray, origin, ray
                            )

                            if midpoint is None:
                                continue

                            candidate_stack[label].append(
                                (midpoint[0], midpoint[1])
                            )
                            total_pairs_triangulated += 1

                        ray_stack[label].append((origin, ray))

                process_count += 1
                self.get_logger().info(
                    f'Frame {frame_count} processed | '
                    f'robot: ({rx:.3f},{ry:.3f})'
                )

            frame_count += 1

        self.get_logger().info(f'Total frames in bag: {frame_count}')
        self.get_logger().info(f'Frames processed: {process_count}')
        self.get_logger().info(f'Pairs triangulated: {total_pairs_triangulated}')
        self.get_logger().info(f'Pairs skipped: {total_pairs_skipped}')

        # DBSCAN clustering -> object stack
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
        json_path = os.path.join(output_dir, 'object_stack.json')
        with open(json_path, 'w') as f:
            json.dump(object_stack, f, indent=2)

        self.get_logger().info(f'Object stack saved to: {json_path}')


def main(args=None):
    rclpy.init(args=args)
    node = FrameProcessor()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
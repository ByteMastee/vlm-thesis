#!/usr/bin/env python3

import time
import numpy as np
import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from tf2_msgs.msg import TFMessage
from sensor_msgs.msg import Image
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import rosbag2_py
import cv2
from ultralytics import YOLO

BAG_PATH = '/root/UVC_ws/vf_robot_model_ros2/thesis_fisheye_bag3'
IMAGE_TOPIC = '/fisheye/front/fisheye_front/image_raw'
ODOM_TOPIC = '/odom'
TF_STATIC_TOPIC = '/tf_static'
TF_TOPIC = '/tf'
FRAME_SKIP = 12
CONFIDENCE = 0.45
MODEL_PATH = '/root/yolo26m.pt'
RAY_LENGTH = 5.0

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


class RayVisualizer(Node):
    def __init__(self):
        super().__init__('ray_visualizing_node')

        self.publisher = self.create_publisher(MarkerArray, '/thesis/rays', 10)
        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)
        self.tf_pub = self.create_publisher(TFMessage, '/tf', 10)
        self.image_pub = self.create_publisher(Image, '/thesis/image_raw', 10)

        tf_static_qos = QoSProfile(
            depth=100,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE
        )
        self.tf_static_pub = self.create_publisher(TFMessage, '/tf_static', tf_static_qos)

        self.get_logger().info('Ray visualizer started')

        model = YOLO(MODEL_PATH)

        storage_options = rosbag2_py.StorageOptions(uri=BAG_PATH, storage_id='sqlite3')
        converter_options = rosbag2_py.ConverterOptions(
            input_serialization_format='cdr',
            output_serialization_format='cdr'
        )

        # First pass: publish tf_static
        tf_reader = rosbag2_py.SequentialReader()
        tf_reader.open(storage_options, converter_options)
        topic_types_list = tf_reader.get_all_topics_and_types()
        type_map = {t.name: t.type for t in topic_types_list}
        tf_filter = rosbag2_py.StorageFilter(topics=[TF_STATIC_TOPIC])
        tf_reader.set_filter(tf_filter)
        tf_msg_type = get_message(type_map[TF_STATIC_TOPIC])
        while tf_reader.has_next():
            topic, data, timestamp = tf_reader.read_next()
            msg = deserialize_message(data, tf_msg_type)
            self.tf_static_pub.publish(msg)
        self.get_logger().info('Published tf_static — waiting for RViz...')

        time.sleep(1.0)

        # Second pass: main processing loop
        reader = rosbag2_py.SequentialReader()
        reader.open(storage_options, converter_options)

        storage_filter = rosbag2_py.StorageFilter(topics=[IMAGE_TOPIC, ODOM_TOPIC, TF_TOPIC])
        reader.set_filter(storage_filter)

        image_msg_type = get_message(type_map[IMAGE_TOPIC])
        odom_msg_type = get_message(type_map[ODOM_TOPIC])
        tf_dynamic_msg_type = get_message(type_map[TF_TOPIC])

        frame_count = 0
        latest_odom = None
        marker_id = 0
        marker_array = MarkerArray()

        while reader.has_next():
            topic, data, timestamp = reader.read_next()

            if topic == TF_TOPIC:
                tf_msg = deserialize_message(data, tf_dynamic_msg_type)
                self.tf_pub.publish(tf_msg)
                continue

            if topic == ODOM_TOPIC:
                latest_odom = deserialize_message(data, odom_msg_type)
                self.odom_pub.publish(latest_odom)
                continue

            if topic != IMAGE_TOPIC:
                continue

            img_msg = deserialize_message(data, image_msg_type)
            self.image_pub.publish(img_msg)

            if frame_count % FRAME_SKIP == 0 and latest_odom is not None:
                img_array = np.frombuffer(bytes(img_msg.data), dtype=np.uint8)

                if img_msg.encoding == 'rgb8':
                    img = img_array.reshape((img_msg.height, img_msg.width, 3))
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
                        end = origin + ray * RAY_LENGTH

                        marker = Marker()
                        marker.header.frame_id = 'odom'
                        marker.header.stamp = self.get_clock().now().to_msg()
                        marker.ns = 'rays'
                        marker.id = marker_id
                        marker.type = Marker.ARROW
                        marker.action = Marker.ADD
                        marker.scale.x = 0.05
                        marker.scale.y = 0.10
                        marker.scale.z = 0.10
                        marker.color.a = 1.0
                        marker.color.r = 1.0
                        marker.color.g = 1.0
                        marker.color.b = 0.0

                        p_start = Point()
                        p_start.x = float(origin[0])
                        p_start.y = float(origin[1])
                        p_start.z = float(origin[2])

                        p_end = Point()
                        p_end.x = float(end[0])
                        p_end.y = float(end[1])
                        p_end.z = float(end[2])

                        marker.points = [p_start, p_end]
                        marker_array.markers.append(marker)
                        marker_id += 1

                        self.get_logger().info(
                            f'Frame {frame_count} | {label} ({cx},{cy}) | '
                            f'origin: ({origin[0]:.3f},{origin[1]:.3f},{origin[2]:.3f}) | '
                            f'ray: ({ray[0]:.3f},{ray[1]:.3f},{ray[2]:.3f})'
                        )

            frame_count += 1

        self.get_logger().info(f'Total markers: {len(marker_array.markers)}')
        self.get_logger().info('Publishing markers on /thesis/rays')

        self.create_timer(1.0, lambda: self.publisher.publish(marker_array))


def main(args=None):
    rclpy.init(args=args)
    node = RayVisualizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
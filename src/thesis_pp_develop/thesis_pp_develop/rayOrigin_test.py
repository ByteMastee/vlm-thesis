#!/usr/bin/env python3

import numpy as np
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import rosbag2_py

BAG_PATH = '/root/UVC_ws/vf_robot_model_ros2/thesis_fisheye_bag3'
IMAGE_TOPIC = '/fisheye/front/fisheye_front/image_raw'
ODOM_TOPIC = '/odom'
FRAME_SKIP = 12

# Translation: base_link -> camera_fisheye_front_optical_frame (from extract_static_tf.py)
# This is T_base_to_optical[:3, 3]
t_base_to_optical = np.array([7.00000000e-02, 0.00000000e+00, 1.84500000e+00])

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

image_msg_type = get_message(topic_types[0].type if False else type_map[IMAGE_TOPIC])
odom_msg_type = get_message(type_map[ODOM_TOPIC])

frame_count = 0
latest_odom = None


def get_rotation_matrix_2d(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]])


while reader.has_next():
    topic, data, timestamp = reader.read_next()

    if topic == ODOM_TOPIC:
        latest_odom = deserialize_message(data, odom_msg_type)
        continue

    if topic != IMAGE_TOPIC:
        continue

    if frame_count % FRAME_SKIP == 0 and latest_odom is not None:
        rx = latest_odom.pose.pose.position.x
        ry = latest_odom.pose.pose.position.y

        q = latest_odom.pose.pose.orientation
        yaw = np.arctan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y**2 + q.z**2))

        # Rotate camera offset from base_link to odom frame (2D)
        R2d = get_rotation_matrix_2d(yaw)
        cam_offset_rotated = R2d @ t_base_to_optical[:2]

        # Camera origin in odom frame
        cam_x = rx + cam_offset_rotated[0]
        cam_y = ry + cam_offset_rotated[1]
        cam_z = t_base_to_optical[2]  # height is constant

        print(f'Frame {frame_count:4d} -> '
              f'robot: ({rx:+.4f}, {ry:+.4f})  '
              f'cam_origin: ({cam_x:+.4f}, {cam_y:+.4f}, {cam_z:+.4f})')

    frame_count += 1
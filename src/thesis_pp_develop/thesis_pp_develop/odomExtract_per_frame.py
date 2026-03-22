#!/usr/bin/env python3

import numpy as np
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import rosbag2_py

BAG_PATH = '/root/UVC_ws/vf_robot_model_ros2/thesis_fisheye_bag3'
IMAGE_TOPIC = '/fisheye/front/fisheye_front/image_raw'
ODOM_TOPIC = '/odom'
FRAME_SKIP = 12

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

frame_count = 0
latest_odom = None
results = []

while reader.has_next():
    topic, data, timestamp = reader.read_next()

    if topic == ODOM_TOPIC:
        msg = deserialize_message(data, odom_msg_type)
        latest_odom = msg
        continue

    if topic != IMAGE_TOPIC:
        continue

    if frame_count % FRAME_SKIP == 0:
        if latest_odom is not None:
            x = latest_odom.pose.pose.position.x
            y = latest_odom.pose.pose.position.y
            z = latest_odom.pose.pose.position.z

            q = latest_odom.pose.pose.orientation
            qx, qy, qz, qw = q.x, q.y, q.z, q.w
            yaw = np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy**2 + qz**2))

            results.append({
                'frame': frame_count,
                'x': round(x, 4),
                'y': round(y, 4),
                'z': round(z, 4),
                'yaw_deg': round(np.degrees(yaw), 3)
            })

            print(f'Frame {frame_count:4d} -> '
                  f'x: {x:+.4f}  y: {y:+.4f}  z: {z:+.4f}  '
                  f'yaw: {np.degrees(yaw):+.3f} deg')
        else:
            print(f'Frame {frame_count}: odom not available yet')

    frame_count += 1

print(f'\nTotal frames with odom: {len(results)}')
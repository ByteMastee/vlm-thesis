#!/usr/bin/env python3

import numpy as np
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import rosbag2_py

BAG_PATH = '/root/UVC_ws/vf_robot_model_ros2/thesis_bag_1'
CAMERA_INFO_TOPIC = '/fisheye/front/fisheye_front/camera_info'

storage_options = rosbag2_py.StorageOptions(uri=BAG_PATH, storage_id='sqlite3')
converter_options = rosbag2_py.ConverterOptions(
    input_serialization_format='cdr',
    output_serialization_format='cdr'
)

reader = rosbag2_py.SequentialReader()
reader.open(storage_options, converter_options)

topic_types = reader.get_all_topics_and_types()
type_map = {t.name: t.type for t in topic_types}

storage_filter = rosbag2_py.StorageFilter(topics=[CAMERA_INFO_TOPIC])
reader.set_filter(storage_filter)

msg_type = get_message(type_map[CAMERA_INFO_TOPIC])

topic, data, timestamp = reader.read_next()
msg = deserialize_message(data, msg_type)

print(f'Width  : {msg.width}')
print(f'Height : {msg.height}')
print(f'K (intrinsics) : {list(msg.k)}')
print(f'D (distortion) : {list(msg.d)}')
print(f'Distortion model: {msg.distortion_model}')
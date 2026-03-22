#!/usr/bin/env python3

import numpy as np
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import rosbag2_py

BAG_PATH = '/root/UVC_ws/vf_robot_model_ros2/thesis_fisheye_bag3'
TF_STATIC_TOPIC = '/tf_static'
SOURCE_FRAME = 'camera_fisheye_front_optical_frame'
TARGET_FRAME = 'base_link'

storage_options = rosbag2_py.StorageOptions(uri=BAG_PATH, storage_id='sqlite3')
converter_options = rosbag2_py.ConverterOptions(
    input_serialization_format='cdr',
    output_serialization_format='cdr'
)

reader = rosbag2_py.SequentialReader()
reader.open(storage_options, converter_options)

topic_types = reader.get_all_topics_and_types()
type_map = {t.name: t.type for t in topic_types}

storage_filter = rosbag2_py.StorageFilter(topics=[TF_STATIC_TOPIC])
reader.set_filter(storage_filter)

msg_type = get_message(type_map[TF_STATIC_TOPIC])

transform_found = None

while reader.has_next():
    topic, data, timestamp = reader.read_next()

    msg = deserialize_message(data, msg_type)

    for tf in msg.transforms:
        print(f'{tf.header.frame_id}  -->  {tf.child_frame_id}')

if transform_found is None:
    print(f'Transform not found between {SOURCE_FRAME} and {TARGET_FRAME}')
else:
    t = transform_found.transform.translation
    q = transform_found.transform.rotation

    print(f'Parent frame : {transform_found.header.frame_id}')
    print(f'Child frame  : {transform_found.child_frame_id}')
    print(f'Translation  : x={t.x:.6f}  y={t.y:.6f}  z={t.z:.6f}')
    print(f'Quaternion   : x={q.x:.6f}  y={q.y:.6f}  z={q.z:.6f}  w={q.w:.6f}')

    # Convert quaternion to rotation matrix
    qx, qy, qz, qw = q.x, q.y, q.z, q.w

    R = np.array([
        [1 - 2*(qy**2 + qz**2),   2*(qx*qy - qz*qw),   2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),   1 - 2*(qx**2 + qz**2),   2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),   2*(qy*qz + qx*qw),   1 - 2*(qx**2 + qy**2)]
    ])

    print(f'\nRotation matrix:\n{R}')
#!/usr/bin/env python3

import numpy as np
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import rosbag2_py

BAG_PATH = '/root/UVC_ws/vf_robot_model_ros2/thesis_fisheye_bag3'
TF_STATIC_TOPIC = '/tf_static'

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

tf_map = {}

while reader.has_next():
    topic, data, timestamp = reader.read_next()
    msg = deserialize_message(data, msg_type)
    for tf in msg.transforms:
        parent = tf.header.frame_id.lstrip('/')
        child = tf.child_frame_id.lstrip('/')
        tf_map[(parent, child)] = tf.transform


def quat_to_rotation_matrix(q):
    qx, qy, qz, qw = q.x, q.y, q.z, q.w
    return np.array([
        [1 - 2*(qy**2 + qz**2),  2*(qx*qy - qz*qw),  2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),  1 - 2*(qx**2 + qz**2),  2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),  2*(qy*qz + qx*qw),  1 - 2*(qx**2 + qy**2)]
    ])


def tf_to_matrix(transform):
    R = quat_to_rotation_matrix(transform.rotation)
    t = np.array([transform.translation.x,
                  transform.translation.y,
                  transform.translation.z])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


# T: base_link -> camera_fisheye_front_link
T1 = tf_to_matrix(tf_map[('base_link', 'camera_fisheye_front_link')])

# T: camera_fisheye_front_link -> camera_fisheye_front_optical_frame
T2 = tf_to_matrix(tf_map[('camera_fisheye_front_link', 'camera_fisheye_front_optical_frame')])

# T: base_link -> camera_fisheye_front_optical_frame
T_base_to_optical = T1 @ T2

# Invert to get: camera_fisheye_front_optical_frame -> base_link
T_optical_to_base = np.linalg.inv(T_base_to_optical)

print('T base_link -> optical_frame:')
print(T_base_to_optical)
print('\nT optical_frame -> base_link:')
print(T_optical_to_base)
print('\nRotation matrix (optical -> base_link):')
print(T_optical_to_base[:3, :3])
print('\nTranslation (optical -> base_link):')
print(T_optical_to_base[:3, 3])
#!/usr/bin/env python3

import cv2
import numpy as np
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import rosbag2_py

BAG_PATH = '/root/UVC_ws/vf_robot_model_ros2/thesis_fisheye_bag4'
IMAGE_TOPIC = '/fisheye/front/fisheye_front/image_raw'
ODOM_TOPIC = '/odom'

FRAME_SKIP = 10  # change this value to test different skip rates

storage_options = rosbag2_py.StorageOptions(uri=BAG_PATH, storage_id='sqlite3')
converter_options = rosbag2_py.ConverterOptions(
    input_serialization_format='cdr',
    output_serialization_format='cdr'
)

reader = rosbag2_py.SequentialReader()
reader.open(storage_options, converter_options)

topic_types = reader.get_all_topics_and_types()
type_map = {t.name: t.type for t in topic_types}

filter = rosbag2_py.StorageFilter(topics=[IMAGE_TOPIC, ODOM_TOPIC])
reader.set_filter(filter)

image_msg_type = get_message(type_map[IMAGE_TOPIC])
odom_msg_type = get_message(type_map[ODOM_TOPIC])

frame_count = 0
latest_odom = None

while reader.has_next():
    topic, data, timestamp = reader.read_next()

    if topic == ODOM_TOPIC:
        latest_odom = deserialize_message(data, odom_msg_type)
        continue

    if topic != IMAGE_TOPIC:
        continue

    msg = deserialize_message(data, image_msg_type)

    if frame_count % FRAME_SKIP == 0:
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
            print(f'Unsupported encoding: {msg.encoding}')
            frame_count += 1
            continue

        cv2.putText(img, f'Frame: {frame_count}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        if latest_odom is not None:
            x = latest_odom.pose.pose.position.x
            y = latest_odom.pose.pose.position.y
            cv2.putText(img, f'Odom x: {x:.3f}  y: {y:.3f}', (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(img, 'Odom: N/A', (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow('Frame Skip Inspector', img)

        key = cv2.waitKey(730)
        if key == ord('q'):
            break

    frame_count += 1

cv2.destroyAllWindows()
print(f'Total frames in bag: {frame_count}')
print(f'Frames viewed (every {FRAME_SKIP}): {frame_count // FRAME_SKIP}')
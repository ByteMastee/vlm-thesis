#!/usr/bin/env python3

import os
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import rosbag2_py


class FrameProcessor(Node):
    def __init__(self):
        super().__init__('frame_processing_node')

        self.declare_parameter('bag_path', '/root/UVC_ws/vf_robot_model_ros2/thesis_fisheye_bag2')
        self.declare_parameter('image_topic', '/fisheye/front/fisheye_front/image_raw')
        self.declare_parameter('output_dir', '/root/UVC_ws/vf_robot_model_ros2/bag_frames')
        self.declare_parameter('frame_skip', 10)

        bag_path = self.get_parameter('bag_path').value
        image_topic = self.get_parameter('image_topic').value
        output_dir = self.get_parameter('output_dir').value
        frame_skip = self.get_parameter('frame_skip').value

        os.makedirs(output_dir, exist_ok=True)

        self.get_logger().info(f'Reading bag: {bag_path}')
        self.get_logger().info(f'Frame skip: {frame_skip}')

        storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id='sqlite3')
        converter_options = rosbag2_py.ConverterOptions(
            input_serialization_format='cdr',
            output_serialization_format='cdr'
        )

        reader = rosbag2_py.SequentialReader()
        reader.open(storage_options, converter_options)

        topic_types = reader.get_all_topics_and_types()
        type_map = {t.name: t.type for t in topic_types}

        storage_filter = rosbag2_py.StorageFilter(topics=[image_topic])
        reader.set_filter(storage_filter)

        msg_type = get_message(type_map[image_topic])

        frame_count = 0
        saved_count = 0

        while reader.has_next():
            topic, data, timestamp = reader.read_next()

            if topic != image_topic:
                continue

            msg = deserialize_message(data, msg_type)

            if frame_count % frame_skip == 0:
                img_array = np.frombuffer(bytes(msg.data), dtype=np.uint8)

                if msg.encoding == 'rgb8':
                    img = img_array.reshape((msg.height, msg.width, 3))
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                elif msg.encoding == 'bgr8':
                    img = img_array.reshape((msg.height, msg.width, 3))
                elif msg.encoding == 'mono8':
                    img = img_array.reshape((msg.height, msg.width))
                else:
                    self.get_logger().warn(f'Unknown encoding: {msg.encoding}')
                    frame_count += 1
                    continue

                filename = os.path.join(output_dir, f'frame_{frame_count:05d}.png')
                cv2.imwrite(filename, img)
                saved_count += 1
                self.get_logger().info(f'Saved frame {frame_count} -> {filename}')

            frame_count += 1

        self.get_logger().info(f'Done. Total frames: {frame_count}, Saved: {saved_count}')


def main(args=None):
    rclpy.init(args=args)
    node = FrameProcessor()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
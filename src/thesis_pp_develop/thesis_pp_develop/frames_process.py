#!/usr/bin/env python3

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
        self.declare_parameter('frame_skip', 7)

        bag_path = self.get_parameter('bag_path').value
        image_topic = self.get_parameter('image_topic').value
        frame_skip = self.get_parameter('frame_skip').value

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
        process_count = 0

        while reader.has_next():
            topic, data, timestamp = reader.read_next()

            if topic != image_topic:
                continue

            if frame_count % frame_skip == 0:
                process_count += 1
                self.get_logger().info(f'Processing frame: {frame_count}')

            frame_count += 1

        self.get_logger().info(f'Total frames in bag: {frame_count}')
        self.get_logger().info(f'Frames to process (every {frame_skip}): {process_count}')


def main(args=None):
    rclpy.init(args=args)
    node = FrameProcessor()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
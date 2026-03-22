#!/usr/bin/env python3

import os
import json
import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import rosbag2_py
from ultralytics import YOLO


class FrameProcessor(Node):
    def __init__(self):
        super().__init__('frame_processing_node')

        self.declare_parameter('bag_path', '/root/UVC_ws/vf_robot_model_ros2/thesis_fisheye_bag3')
        self.declare_parameter('image_topic', '/fisheye/front/fisheye_front/image_raw')
        self.declare_parameter('frame_skip', 12)
        self.declare_parameter('confidence', 0.45)
        self.declare_parameter('model_path', '/root/yolo26m.pt')
        self.declare_parameter('output_dir', '/root/UVC_ws/vf_robot_model_ros2/step5_detections')

        bag_path = self.get_parameter('bag_path').value
        image_topic = self.get_parameter('image_topic').value
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

        storage_filter = rosbag2_py.StorageFilter(topics=[image_topic])
        reader.set_filter(storage_filter)

        msg_type = get_message(type_map[image_topic])

        frame_count = 0
        process_count = 0
        all_detections = []

        while reader.has_next():
            topic, data, timestamp = reader.read_next()

            if topic != image_topic:
                continue

            if frame_count % frame_skip == 0:
                msg = deserialize_message(data, msg_type)
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

                frame_detections = []

                for result in results:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cls_id = int(box.cls[0])
                        label = model.names[cls_id]
                        cx = (x1 + x2) // 2
                        cy = (y1 + y2) // 2

                        frame_detections.append({
                            'label': label,
                            'centroid_px': [cx, cy]
                        })

                if frame_detections:
                    all_detections.extend(frame_detections)

                process_count += 1
                self.get_logger().info(f'Frame {frame_count}: {len(frame_detections)} detections')

            frame_count += 1

        json_path = os.path.join(output_dir, 'detections.json')
        with open(json_path, 'w') as f:
            json.dump(all_detections, f, indent=2)

        self.get_logger().info(f'Total frames in bag: {frame_count}')
        self.get_logger().info(f'Frames processed: {process_count}')
        self.get_logger().info(f'Detections saved to: {json_path}')


def main(args=None):
    rclpy.init(args=args)
    node = FrameProcessor()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
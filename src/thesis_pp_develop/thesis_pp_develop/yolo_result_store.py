#!/usr/bin/env python3

import os
import json
import cv2
import numpy as np
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import rosbag2_py
from ultralytics import YOLO

BAG_PATH = '/root/UVC_ws/vf_robot_model_ros2/thesis_fisheye_bag3'
IMAGE_TOPIC = '/fisheye/front/fisheye_front/image_raw'
OUTPUT_DIR = '/root/UVC_ws/vf_robot_model_ros2/yolo_detections'
FRAME_SKIP = 12
CONFIDENCE = 0.45
MODEL = '/root/yolo26m.pt'

os.makedirs(OUTPUT_DIR, exist_ok=True)

model = YOLO(MODEL)

storage_options = rosbag2_py.StorageOptions(uri=BAG_PATH, storage_id='sqlite3')
converter_options = rosbag2_py.ConverterOptions(
    input_serialization_format='cdr',
    output_serialization_format='cdr'
)

reader = rosbag2_py.SequentialReader()
reader.open(storage_options, converter_options)

topic_types = reader.get_all_topics_and_types()
type_map = {t.name: t.type for t in topic_types}

storage_filter = rosbag2_py.StorageFilter(topics=[IMAGE_TOPIC])
reader.set_filter(storage_filter)

msg_type = get_message(type_map[IMAGE_TOPIC])

frame_count = 0
all_detections = []

while reader.has_next():
    topic, data, timestamp = reader.read_next()

    if topic != IMAGE_TOPIC:
        continue

    if frame_count % FRAME_SKIP == 0:
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
            print(f'Unsupported encoding: {msg.encoding}')
            frame_count += 1
            continue

        results = model(img, conf=CONFIDENCE, verbose=False)

        frame_detections = []

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                label = model.names[cls_id]

                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                detection = {
                    'label': label,
                    'confidence': round(conf, 3),
                    'centroid_px': [cx, cy],
                    'bbox': [x1, y1, x2, y2]
                }

                frame_detections.append(detection)

        frame_entry = {
            'frame': frame_count,
            'timestamp': timestamp,
            'detections': frame_detections
        }

        all_detections.append(frame_entry)

        print(f'Frame {frame_count}: {len(frame_detections)} detections')

    frame_count += 1

# Save as JSON
json_path = os.path.join(OUTPUT_DIR, 'detections2.json')
with open(json_path, 'w') as f:
    json.dump(all_detections, f, indent=2)

print(f'\nTotal frames processed: {len(all_detections)}')
print(f'Saved detections to: {json_path}')
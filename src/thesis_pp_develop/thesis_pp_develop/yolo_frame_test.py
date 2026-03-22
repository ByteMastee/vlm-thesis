#!/usr/bin/env python3

import os
import cv2
import numpy as np
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import rosbag2_py
from ultralytics import YOLO

BAG_PATH = '/root/UVC_ws/vf_robot_model_ros2/thesis_fisheye_bag2'
IMAGE_TOPIC = '/fisheye/front/fisheye_front/image_raw'
FRAME_SKIP = 7
CONFIDENCE = 0.5
MODEL = '/root/yolo26n.pt'
OUTPUT_DIR = '/root/UVC_ws/vf_robot_model_ros2/yolo_frames2'

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

while reader.has_next():
    topic, data, timestamp = reader.read_next()

    if topic != IMAGE_TOPIC:
        continue

    if frame_count % FRAME_SKIP == 0:
        msg = deserialize_message(data, msg_type)
        img_array = np.frombuffer(bytes(msg.data), dtype=np.uint8)

        #Just for debugging, print encoding and dimensions
        #print(f'Encoding: {msg.encoding}, Shape: {msg.height}x{msg.width}')

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

        #Just for debugging, print number of detections in this frame
        # total_detections = sum(len(r.boxes) for r in results)
        # print(f'Frame {frame_count}: {total_detections} detections')

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                label = model.names[cls_id]

                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                # Bounding box
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Centroid
                cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1)

                # Label and confidence
                cv2.putText(img, f'{label} {conf:.2f}', (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Centroid coordinates
                cv2.putText(img, f'({cx},{cy})', (cx + 6, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        cv2.putText(img, f'Frame: {frame_count}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        cv2.imshow('YOLO Frame Test', img)

        cv2.imwrite(os.path.join(OUTPUT_DIR, f'frame_{frame_count:05d}.png'), img)

        key = cv2.waitKey(450)
        if key == ord('q'):
            break

    frame_count += 1

cv2.destroyAllWindows()
print(f'Total frames: {frame_count}')
#!/usr/bin/env python3

import cv2
import numpy as np
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import rosbag2_py
from ultralytics import YOLO

BAG_PATH = '/root/UVC_ws/vf_robot_model_ros2/thesis_fisheye_bag3'
IMAGE_TOPIC = '/fisheye/front/fisheye_front/image_raw'
ODOM_TOPIC = '/odom'
FRAME_SKIP = 12
CONFIDENCE = 0.45
MODEL_PATH = '/root/yolo26m.pt'

# Camera intrinsics
FX = 28.00600204423685
FY = 28.00600204423685
CX_0 = 320.5
CY_0 = 240.5

# Rotation matrix: optical_frame -> base_link
R_optical_to_base = np.array([
    [-2.55002079e-02, -9.99674817e-01,  2.15810911e-06],
    [-5.87339488e-01,  1.49804041e-02, -8.09202023e-01],
    [ 8.08938852e-01, -2.06360874e-02, -5.87530498e-01]
])


def pixel_to_ray_camera(cx, cy):
    x = (cx - CX_0) / FX
    y = (cy - CY_0) / FY
    z = 1.0
    ray = np.array([x, y, z])
    ray = ray / np.linalg.norm(ray)
    return ray


def ray_to_image_arrow(cx, cy, ray_cam, scale=60):
    # Project ray back onto image plane for visualization
    # Use x and y components of ray in camera frame as 2D direction
    dx = ray_cam[0] * scale
    dy = ray_cam[1] * scale
    end_x = int(cx + dx)
    end_y = int(cy + dy)
    return end_x, end_y

#If need to see from principal point, use the below:
# def ray_to_image_arrow(cx, cy):
#     # Ray from principal point (camera center) to detected centroid
#     return cx, cy

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

model = YOLO(MODEL_PATH)

frame_count = 0
latest_odom = None

while reader.has_next():
    topic, data, timestamp = reader.read_next()

    if topic == ODOM_TOPIC:
        latest_odom = deserialize_message(data, odom_msg_type)
        continue

    if topic != IMAGE_TOPIC:
        continue

    if frame_count % FRAME_SKIP == 0:
        msg = deserialize_message(data, image_msg_type)
        img_array = np.frombuffer(bytes(msg.data), dtype=np.uint8)

        if msg.encoding == 'rgb8':
            img = img_array.reshape((msg.height, msg.width, 3))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            frame_count += 1
            continue

        results = model(img, conf=CONFIDENCE, verbose=False)

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                # Bounding box
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Centroid
                cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1)

                # Label
                cv2.putText(img, label, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Ray in camera frame
                ray_cam = pixel_to_ray_camera(cx, cy)

                # Project ray onto image as arrow
                ex, ey = ray_to_image_arrow(cx, cy, ray_cam, scale=60)
                cv2.arrowedLine(img, (cx, cy), (ex, ey), (255, 0, 0), 2, tipLength=0.3)

                #If need to see from principal point, use the below:
                principal_x, principal_y = int(CX_0), int(CY_0)
                cv2.arrowedLine(img, (principal_x, principal_y), (cx, cy), (255, 0, 0), 2, tipLength=0.3)

                # Print ray info
                ray_base = R_optical_to_base @ ray_cam
                ray_base = ray_base / np.linalg.norm(ray_base)
                cv2.putText(img,
                            f'ray_base: ({ray_base[0]:.2f},{ray_base[1]:.2f},{ray_base[2]:.2f})',
                            (x1, y2 + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1)

        # Odom overlay
        if latest_odom is not None:
            x = latest_odom.pose.pose.position.x
            y = latest_odom.pose.pose.position.y
            q = latest_odom.pose.pose.orientation
            yaw = np.degrees(np.arctan2(
                2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y**2 + q.z**2)))
            cv2.putText(img, f'Frame: {frame_count}', (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(img, f'x:{x:.3f} y:{y:.3f} yaw:{yaw:.1f}deg', (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.imshow('Ray Image Verify', img)
        key = cv2.waitKey(0)  # wait for keypress to go to next frame
        if key == ord('q'):
            break

    frame_count += 1

cv2.destroyAllWindows()
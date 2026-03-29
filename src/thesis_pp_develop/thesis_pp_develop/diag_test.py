#!/usr/bin/env python3

import numpy as np
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import rosbag2_py
import cv2
from ultralytics import YOLO

BAG_PATH = '/root/UVC_ws/vf_robot_model_ros2/diag_bag'
IMAGE_TOPIC = '/fisheye/front/fisheye_front/image_raw'
ODOM_TOPIC = '/odom'

FX = 554.26
FY = 554.26
CX_0 = 320.5
CY_0 = 240.5

R_optical_to_base = np.array([
    [-2.55002079e-02, -9.99674817e-01,  2.15810911e-06],
    [-5.87339488e-01,  1.49804041e-02, -8.09202023e-01],
    [ 8.08938852e-01, -2.06360874e-02, -5.87530498e-01]
])

T_BASE_TO_CAM = np.array([0.07, 0.0, 1.845])

model = YOLO('/root/yolo26m.pt')

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
odom_msg_type  = get_message(type_map[ODOM_TOPIC])

latest_odom = None
frame_count  = 0

while reader.has_next():
    topic, data, timestamp = reader.read_next()

    if topic == ODOM_TOPIC:
        latest_odom = deserialize_message(data, odom_msg_type)
        continue

    if topic != IMAGE_TOPIC or latest_odom is None:
        continue

    msg = deserialize_message(data, image_msg_type)
    img_array = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    img = img_array.reshape((msg.height, msg.width, 3))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    rx  = latest_odom.pose.pose.position.x
    ry  = latest_odom.pose.pose.position.y
    q   = latest_odom.pose.pose.orientation
    yaw = np.arctan2(
        2 * (q.w * q.z + q.x * q.y),
        1 - 2 * (q.y**2 + q.z**2)
    )

    results = model(img, conf=0.3, verbose=False)

    # Save every frame regardless of detection
    annotated = results[0].plot()
    save_path = f'/root/UVC_ws/vf_robot_model_ros2/diagnostic/diag_frame_{frame_count:03d}.png'
    cv2.imwrite(save_path, annotated)

    num_detections = sum(len(r.boxes) for r in results)
    print(f'Frame {frame_count} | robot: ({rx:.3f},{ry:.3f}) | yaw: {np.degrees(yaw):.1f}° | detections: {num_detections}')

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            label  = model.names[cls_id]
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            x_n = (cx - CX_0) / FX
            y_n = (cy - CY_0) / FY
            ray_cam = np.array([x_n, y_n, 1.0])
            ray_cam = ray_cam / np.linalg.norm(ray_cam)

            ray_base = R_optical_to_base @ ray_cam
            ray_base = ray_base / np.linalg.norm(ray_base)

            c, s = np.cos(yaw), np.sin(yaw)
            R_yaw = np.array([[c,-s,0],[s,c,0],[0,0,1]])
            ray_odom = R_yaw @ ray_base
            ray_odom = ray_odom / np.linalg.norm(ray_odom)

            R2d = np.array([[c,-s],[s,c]])
            cam_offset_rotated = R2d @ T_BASE_TO_CAM[:2]
            cam_origin = np.array([
                rx + cam_offset_rotated[0],
                ry + cam_offset_rotated[1],
                T_BASE_TO_CAM[2]
            ])

            gt = np.array([-3.0, 2.0, 0.0])
            gt_vec = gt - cam_origin
            proj = np.dot(gt_vec, ray_odom)
            proj_point = cam_origin + proj * ray_odom

            print(f'  {label} | centroid: ({cx},{cy}) | '
                  f'ray_odom: ({ray_odom[0]:.3f},{ray_odom[1]:.3f},{ray_odom[2]:.3f})')
            print(f'  cam_origin: ({cam_origin[0]:.3f},{cam_origin[1]:.3f})')
            print(f'  proj on ray: ({proj_point[0]:.3f},{proj_point[1]:.3f})')
            print(f'  offset from GT: ({proj_point[0]-gt[0]:.3f},{proj_point[1]-gt[1]:.3f})')

    frame_count += 1

print(f'\nTotal frames: {frame_count}')
print('Frames saved to pp_tunning/diag_frame_XXX.png')
#!/usr/bin/env python3

import json
import numpy as np
import matplotlib.pyplot as plt
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import rosbag2_py

OBJECT_STACK_PATH = '/root/UVC_ws/vf_robot_model_ros2/step7_detections/object_stack_tuned.json'
OUTPUT_PATH = '/root/UVC_ws/vf_robot_model_ros2/step7_detections/map_plot_tune1_3.png'
BAG_PATH = '/root/UVC_ws/vf_robot_model_ros2/thesis_fisheye_bag3'
ODOM_TOPIC = '/odom'

# Ground truth from Gazebo world file
ground_truth = {
    'chair_1': (-3.0,  2.0),
    'chair_2': (-3.5, -2.5),
    'couch':   ( 3.5,  0.0),
    'table':   ( 2.0,  2.5)
}

with open(OBJECT_STACK_PATH, 'r') as f:
    object_stack = json.load(f)

# Read odom path from bag
storage_options = rosbag2_py.StorageOptions(uri=BAG_PATH, storage_id='sqlite3')
converter_options = rosbag2_py.ConverterOptions(
    input_serialization_format='cdr',
    output_serialization_format='cdr'
)
reader = rosbag2_py.SequentialReader()
reader.open(storage_options, converter_options)
topic_types = reader.get_all_topics_and_types()
type_map = {t.name: t.type for t in topic_types}
storage_filter = rosbag2_py.StorageFilter(topics=[ODOM_TOPIC])
reader.set_filter(storage_filter)
odom_msg_type = get_message(type_map[ODOM_TOPIC])
robot_x, robot_y = [], []
while reader.has_next():
    topic, data, timestamp = reader.read_next()
    msg = deserialize_message(data, odom_msg_type)
    robot_x.append(msg.pose.pose.position.x)
    robot_y.append(msg.pose.pose.position.y)

plt.figure(figsize=(10, 10))

# Plot robot trajectory
plt.plot(robot_x, robot_y, 'b-', linewidth=1.0, alpha=0.5)
plt.plot(robot_x[0], robot_y[0], 'go', markersize=8)
plt.plot(robot_x[-1], robot_y[-1], 'rs', markersize=8)

# Plot ground truth
for label, (gx, gy) in ground_truth.items():
    plt.plot(gx, gy, 'g^', markersize=12)
    plt.annotate(f'GT: {label}\n({gx},{gy})', (gx, gy),
                 textcoords='offset points', xytext=(8, 8),
                 fontsize=9, color='green')

# Plot detected object stack
colors = ['red', 'orange', 'purple', 'cyan', 'magenta']
for i, (label, data) in enumerate(object_stack.items()):
    ox = data['x']
    oy = data['y']
    color = colors[i % len(colors)]
    plt.plot(ox, oy, '*', markersize=15, color=color)
    plt.annotate(f'Det: {label}\n({ox:.2f},{oy:.2f})', (ox, oy),
                 textcoords='offset points', xytext=(8, -18),
                 fontsize=9, color=color)

    # Draw error line to nearest GT
    best_dist = float('inf')
    best_gx, best_gy = None, None
    for gt_label, (gx, gy) in ground_truth.items():
        dist = np.sqrt((ox - gx)**2 + (oy - gy)**2)
        if dist < best_dist:
            best_dist = dist
            best_gx, best_gy = gx, gy

    if best_gx is not None:
        plt.plot([ox, best_gx], [oy, best_gy], '--', color=color, linewidth=1.0)
        plt.text((ox + best_gx) / 2, (oy + best_gy) / 2,
                 f'{best_dist:.2f}m', fontsize=8, color=color)

plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.title('Semantic Map — Detected vs Ground Truth')
plt.legend(handles=[
    plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='green',
               markersize=10, label='Ground Truth'),
    plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='red',
               markersize=10, label='Detected'),
    plt.Line2D([0], [0], color='blue', linewidth=1.0, label='Robot path'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green',
               markersize=8, label='Start'),
    plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='red',
               markersize=8, label='End')
])
plt.grid(True)
plt.axis('equal')
plt.savefig(OUTPUT_PATH, dpi=150)
plt.show()
print(f'Plot saved to: {OUTPUT_PATH}')
#!/usr/bin/env python3

import json
import numpy as np
import matplotlib.pyplot as plt
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import rosbag2_py

BAG_PATH = '/root/UVC_ws/vf_robot_model_ros2/thesis_fisheye_bag3'
ODOM_TOPIC = '/odom'
OBJECT_STACK_PATH = '/root/UVC_ws/vf_robot_model_ros2/step7_detections/object_stack2.json'

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

robot_x = []
robot_y = []

while reader.has_next():
    topic, data, timestamp = reader.read_next()
    msg = deserialize_message(data, odom_msg_type)
    robot_x.append(msg.pose.pose.position.x)
    robot_y.append(msg.pose.pose.position.y)

with open(OBJECT_STACK_PATH, 'r') as f:
    object_stack = json.load(f)

plt.figure(figsize=(10, 10))
plt.plot(robot_x, robot_y, 'b-', linewidth=1.0, label='Robot path')
plt.plot(robot_x[0], robot_y[0], 'go', markersize=10, label='Start')
plt.plot(robot_x[-1], robot_y[-1], 'rs', markersize=10, label='End')

colors = ['red', 'orange', 'purple', 'cyan', 'magenta']
for i, (label, data) in enumerate(object_stack.items()):
    ox = data['x']
    oy = data['y']
    color = colors[i % len(colors)]
    plt.plot(ox, oy, '*', markersize=15, color=color, label=f'{label} ({ox:.2f},{oy:.2f})')
    plt.annotate(label, (ox, oy), textcoords='offset points',
                 xytext=(8, 8), fontsize=10, color=color)

plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.title('Robot Path + Object Stack')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.savefig('/root/UVC_ws/vf_robot_model_ros2/step7_detections/map_plot.png', dpi=150)
plt.show()
print('Plot saved.')
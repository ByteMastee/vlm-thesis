#!/usr/bin/env python3

import json
import os

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from builtin_interfaces.msg import Duration
from tf2_ros import StaticTransformBroadcaster
from geometry_msgs.msg import TransformStamped

GROUND_TRUTH = {
    'chair_1': (-3.0,  2.0),
    'chair_2': (-3.5, -2.5),
    'couch':   ( 3.5,  0.0),
    'table':   ( 2.0,  2.5),
}


class RvizMapPublisher(Node):
    def __init__(self):
        super().__init__('rviz_map_publish_node')

        self.declare_parameter('object_stack_path',
                               '/root/UVC_ws/vf_robot_model_ros2/pp_tunning/object_stack5.json')
        self.declare_parameter('robot_odom_path',
                               '/root/UVC_ws/vf_robot_model_ros2/pp_tunning/robot_path5.json')

        self.static_broadcaster = StaticTransformBroadcaster(self)

        map_to_odom = TransformStamped()
        map_to_odom.header.stamp = self.get_clock().now().to_msg()
        map_to_odom.header.frame_id = 'map'
        map_to_odom.child_frame_id = 'odom'
        map_to_odom.transform.translation.x = 0.0
        map_to_odom.transform.translation.y = 0.0
        map_to_odom.transform.translation.z = 0.0
        map_to_odom.transform.rotation.w = 1.0
        self.static_broadcaster.sendTransform(map_to_odom)

        object_stack_path = self.get_parameter('object_stack_path').value
        robot_odom_path   = self.get_parameter('robot_odom_path').value

        # Latched QoS — RViz receives markers even if subscribed after publishing
        latched_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )

        self.pub = self.create_publisher(MarkerArray, '/semantic_map_markers', latched_qos)

        # Load object stack
        if not os.path.exists(object_stack_path):
            self.get_logger().error(f'object_stack not found: {object_stack_path}')
            return

        with open(object_stack_path, 'r') as f:
            object_stack = json.load(f)

        # Load robot path (optional)
        robot_path = None
        if os.path.exists(robot_odom_path):
            with open(robot_odom_path, 'r') as f:
                robot_path = json.load(f)  # expects {"x": [...], "y": [...]}
        else:
            self.get_logger().warn(f'robot_path.json not found: {robot_odom_path} — skipping trajectory')

        marker_array = MarkerArray()
        marker_id = 0

        # --- Detected objects ---
        for label, data in object_stack.items():
            ox = data['x']
            oy = data['y']

            # Sphere marker
            sphere = Marker()
            sphere.header.frame_id = 'odom'
            sphere.header.stamp = self.get_clock().now().to_msg()
            sphere.ns = 'detected'
            sphere.id = marker_id
            sphere.type = Marker.SPHERE
            sphere.action = Marker.ADD
            sphere.pose.position.x = ox
            sphere.pose.position.y = oy
            sphere.pose.position.z = 0.0
            sphere.pose.orientation.w = 1.0
            sphere.scale.x = 0.2
            sphere.scale.y = 0.2
            sphere.scale.z = 0.2
            sphere.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)
            sphere.lifetime = Duration(sec=0)
            marker_array.markers.append(sphere)
            marker_id += 1

            # Text label
            text = Marker()
            text.header.frame_id = 'odom'
            text.header.stamp = self.get_clock().now().to_msg()
            text.ns = 'detected_labels'
            text.id = marker_id
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position.x = ox
            text.pose.position.y = oy
            text.pose.position.z = 0.3
            text.pose.orientation.w = 1.0
            text.scale.z = 0.2
            text.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)
            text.text = f'Det: {label}\n({ox:.2f},{oy:.2f})'
            text.lifetime = Duration(sec=0)
            marker_array.markers.append(text)
            marker_id += 1

        # --- Ground truth objects ---
        for label, (gx, gy) in GROUND_TRUTH.items():

            # Sphere marker
            sphere = Marker()
            sphere.header.frame_id = 'odom'
            sphere.header.stamp = self.get_clock().now().to_msg()
            sphere.ns = 'ground_truth'
            sphere.id = marker_id
            sphere.type = Marker.SPHERE
            sphere.action = Marker.ADD
            sphere.pose.position.x = gx
            sphere.pose.position.y = gy
            sphere.pose.position.z = 0.0
            sphere.pose.orientation.w = 1.0
            sphere.scale.x = 0.2
            sphere.scale.y = 0.2
            sphere.scale.z = 0.2
            sphere.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)
            sphere.lifetime = Duration(sec=0)
            marker_array.markers.append(sphere)
            marker_id += 1

            # Text label
            text = Marker()
            text.header.frame_id = 'odom'
            text.header.stamp = self.get_clock().now().to_msg()
            text.ns = 'gt_labels'
            text.id = marker_id
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position.x = gx
            text.pose.position.y = gy
            text.pose.position.z = 0.3
            text.pose.orientation.w = 1.0
            text.scale.z = 0.2
            text.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)
            text.text = f'GT: {label}\n({gx},{gy})'
            text.lifetime = Duration(sec=0)
            marker_array.markers.append(text)
            marker_id += 1

        # --- Robot trajectory ---
        if robot_path is not None:
            traj = Marker()
            traj.header.frame_id = 'odom'
            traj.header.stamp = self.get_clock().now().to_msg()
            traj.ns = 'trajectory'
            traj.id = marker_id
            traj.type = Marker.LINE_STRIP
            traj.action = Marker.ADD
            traj.scale.x = 0.05
            traj.color = ColorRGBA(r=0.0, g=0.5, b=1.0, a=0.8)
            traj.pose.orientation.w = 1.0
            traj.lifetime = Duration(sec=0)

            for px, py in zip(robot_path['x'], robot_path['y']):
                pt = Point()
                pt.x = px
                pt.y = py
                pt.z = 0.0
                traj.points.append(pt)

            marker_array.markers.append(traj)
            marker_id += 1

        self.pub.publish(marker_array)
        self.get_logger().info(f'Published {len(marker_array.markers)} markers to /semantic_map_markers')


def main(args=None):
    rclpy.init(args=args)
    node = RvizMapPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
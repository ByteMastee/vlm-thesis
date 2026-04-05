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


RUN_COLORS = {
    'run1': ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0),
    'run2': ColorRGBA(r=1.0, g=0.5, b=0.0, a=1.0),
    'run3': ColorRGBA(r=0.5, g=0.0, b=1.0, a=1.0),
}

PATH_COLORS = {
    'run1': ColorRGBA(r=1.0, g=0.3, b=0.3, a=0.8),
    'run2': ColorRGBA(r=1.0, g=0.7, b=0.2, a=0.8),
    'run3': ColorRGBA(r=0.7, g=0.3, b=1.0, a=0.8),
}

GT_COLOR   = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)


class MultiRunVisualizer(Node):
    def __init__(self):
        super().__init__('multi_run_visualizer')

        # --- Parameters ---
        self.declare_parameter('object_stack_run1', '')
        self.declare_parameter('object_stack_run2', '')
        self.declare_parameter('object_stack_run3', '')
        self.declare_parameter('robot_path_run1',   '')
        self.declare_parameter('robot_path_run2',   '')
        self.declare_parameter('robot_path_run3',   '')
        self.declare_parameter('ground_truth',       ['chair_1:-3.0:2.0', 'chair_2:-3.5:-2.5', 'couch:3.5:0.0', 'table:2.0:2.5'])

        self.object_stack_paths = {
            'run1': self.get_parameter('object_stack_run1').value,
            'run2': self.get_parameter('object_stack_run2').value,
            'run3': self.get_parameter('object_stack_run3').value,
        }
        self.robot_path_paths = {
            'run1': self.get_parameter('robot_path_run1').value,
            'run2': self.get_parameter('robot_path_run2').value,
            'run3': self.get_parameter('robot_path_run3').value,
        }

        gt_raw = self.get_parameter('ground_truth').value
        self.ground_truth = {}
        for entry in gt_raw:
            parts = entry.split(':')
            self.ground_truth[parts[0]] = (float(parts[1]), float(parts[2]))

        # --- Static TF: map -> odom ---
        self.static_broadcaster = StaticTransformBroadcaster(self)
        map_to_odom                             = TransformStamped()
        map_to_odom.header.stamp                = self.get_clock().now().to_msg()
        map_to_odom.header.frame_id             = 'map'
        map_to_odom.child_frame_id              = 'odom'
        map_to_odom.transform.translation.x     = 0.0
        map_to_odom.transform.translation.y     = 0.0
        map_to_odom.transform.translation.z     = 0.0
        map_to_odom.transform.rotation.w        = 1.0
        self.static_broadcaster.sendTransform(map_to_odom)

        # --- Publisher ---
        latched_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )
        self.pub = self.create_publisher(MarkerArray, '/multi_run_markers', latched_qos)

        # --- Build and publish ---
        marker_array = self._build_marker_array()
        self.pub.publish(marker_array)
        self.get_logger().info(
            f'Published {len(marker_array.markers)} markers to /multi_run_markers'
        )

    def _build_marker_array(self):
        marker_array = MarkerArray()
        marker_id    = 0

        # --- GT markers ---
        for label, (gx, gy) in self.ground_truth.items():
            marker_array.markers.append(self._make_sphere(
                marker_id, gx, gy, 'gt', GT_COLOR, scale=0.25
            ))
            marker_id += 1
            marker_array.markers.append(self._make_text(
                marker_id, gx, gy, f'GT: {label}\n({gx},{gy})', 'gt_labels', GT_COLOR
            ))
            marker_id += 1

        # --- Per run markers ---
        for run_name in ['run1', 'run2', 'run3']:
            obj_path  = self.object_stack_paths[run_name]
            path_path = self.robot_path_paths[run_name]

            if not obj_path or not os.path.exists(obj_path):
                self.get_logger().warn(f'{run_name} object stack not found: {obj_path}')
            else:
                with open(obj_path, 'r') as f:
                    object_stack = json.load(f)

                for label, data in object_stack.items():
                    ox = data['x']
                    oy = data['y']
                    marker_array.markers.append(self._make_sphere(
                        marker_id, ox, oy,
                        f'{run_name}_detected',
                        RUN_COLORS[run_name], scale=0.2
                    ))
                    marker_id += 1
                    marker_array.markers.append(self._make_text(
                        marker_id, ox, oy,
                        f'{run_name}: {label}\n({ox:.2f},{oy:.2f})',
                        f'{run_name}_labels',
                        RUN_COLORS[run_name]
                    ))
                    marker_id += 1

            if not path_path or not os.path.exists(path_path):
                self.get_logger().warn(f'{run_name} robot path not found: {path_path}')
            else:
                with open(path_path, 'r') as f:
                    robot_path = json.load(f)

                traj                    = Marker()
                traj.header.frame_id    = 'odom'
                traj.header.stamp       = self.get_clock().now().to_msg()
                traj.ns                 = f'{run_name}_path'
                traj.id                 = marker_id
                traj.type               = Marker.LINE_STRIP
                traj.action             = Marker.ADD
                traj.scale.x            = 0.05
                traj.color              = PATH_COLORS[run_name]
                traj.pose.orientation.w = 1.0
                traj.lifetime           = Duration(sec=0)

                for px, py in zip(robot_path['x'], robot_path['y']):
                    pt   = Point()
                    pt.x = float(px)
                    pt.y = float(py)
                    pt.z = 0.0
                    traj.points.append(pt)

                marker_array.markers.append(traj)
                marker_id += 1

                # Run label at path start
                if robot_path['x']:
                    marker_array.markers.append(self._make_text(
                        marker_id,
                        robot_path['x'][0],
                        robot_path['y'][0],
                        run_name,
                        f'{run_name}_path_label',
                        PATH_COLORS[run_name],
                        z_offset=0.5
                    ))
                    marker_id += 1

        return marker_array

    # --- Helpers ---

    def _make_sphere(self, marker_id, x, y, ns, color, scale=0.2):
        m                    = Marker()
        m.header.frame_id    = 'odom'
        m.header.stamp       = self.get_clock().now().to_msg()
        m.ns                 = ns
        m.id                 = marker_id
        m.type               = Marker.SPHERE
        m.action             = Marker.ADD
        m.pose.position.x    = float(x)
        m.pose.position.y    = float(y)
        m.pose.position.z    = 0.0
        m.pose.orientation.w = 1.0
        m.scale.x            = scale
        m.scale.y            = scale
        m.scale.z            = scale
        m.color              = color
        m.lifetime           = Duration(sec=0)
        return m

    def _make_text(self, marker_id, x, y, text, ns, color, z_offset=0.3):
        m                    = Marker()
        m.header.frame_id    = 'odom'
        m.header.stamp       = self.get_clock().now().to_msg()
        m.ns                 = ns
        m.id                 = marker_id
        m.type               = Marker.TEXT_VIEW_FACING
        m.action             = Marker.ADD
        m.pose.position.x    = float(x)
        m.pose.position.y    = float(y)
        m.pose.position.z    = z_offset
        m.pose.orientation.w = 1.0
        m.scale.z            = 0.2
        m.color              = color
        m.text               = text
        m.lifetime           = Duration(sec=0)
        return m


def main(args=None):
    rclpy.init(args=args)
    node = MultiRunVisualizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()




# COMMAND TO RUN: CAN be EDITED AS NEEDED

# ros2 run semantic_mapping multi_run_visualizer --ros-args \
#   -p object_stack_run1:=/root/UVC_ws/vf_robot_model_ros2/semantic_mapping_output/object_stack_run1.json \
#   -p robot_path_run1:=/root/UVC_ws/vf_robot_model_ros2/semantic_mapping_output/robot_path_run1.json \
#   -p object_stack_run2:=/root/UVC_ws/vf_robot_model_ros2/semantic_mapping_output/object_stack_run2.json \
#   -p robot_path_run2:=/root/UVC_ws/vf_robot_model_ros2/semantic_mapping_output/robot_path_run2.json \
#   -p object_stack_run3:=/root/UVC_ws/vf_robot_model_ros2/semantic_mapping_output/object_stack_run3.json \
#   -p robot_path_run3:=/root/UVC_ws/vf_robot_model_ros2/semantic_mapping_output/robot_path_run3.json
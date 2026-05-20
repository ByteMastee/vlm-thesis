import os
import json

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from visualization_msgs.msg import MarkerArray

from multi_cam_perception.rviz_publisher_node import RvizPublisherNode

from tf2_ros import StaticTransformBroadcaster
from geometry_msgs.msg import TransformStamped

# --- Run name: must match ros_node.py ---
RUN_NAME = 'run_01'

BASE_OUTPUT_DIR = '/root/UVC_ws/vf_robot_model_ros2/Final_Output/Testing'


class VlmVizNode(Node):
    def __init__(self):
        super().__init__('vlm_rviz_node')

        # --- Parameters ---
        self.declare_parameter('output_dir',   os.path.join(BASE_OUTPUT_DIR, RUN_NAME))
        self.declare_parameter('ground_truth', ['chair_1:-3.0:2.0', 'chair_2:-3.5:-2.5', 'couch:3.5:0.0', 'table:2.0:2.5'])

        # --- Static TF: map -> odom ---
        self._static_broadcaster = StaticTransformBroadcaster(self)
        tf_msg                             = TransformStamped()
        tf_msg.header.stamp                = self.get_clock().now().to_msg()
        tf_msg.header.frame_id             = 'map'
        tf_msg.child_frame_id              = 'odom'
        tf_msg.transform.translation.x     = 0.0
        tf_msg.transform.translation.y     = 0.0
        tf_msg.transform.translation.z     = 0.0
        tf_msg.transform.rotation.w        = 1.0
        self._static_broadcaster.sendTransform(tf_msg)
        self.get_logger().info(f'[{RUN_NAME}] Static TF map -> odom published.')

        self.output_dir = self.get_parameter('output_dir').value
        gt_raw          = self.get_parameter('ground_truth').value

        self.ground_truth = {}
        for entry in gt_raw:
            parts = entry.split(':')
            self.ground_truth[parts[0]] = (float(parts[1]), float(parts[2]))

        # --- Latched QoS ---
        latched_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )

        # --- Publishers ---
        self.vlm_pub = self.create_publisher(MarkerArray, '/vlm_semantic_map_markers', latched_qos)
        self.gt_pub  = self.create_publisher(MarkerArray, '/gt_markers',               latched_qos)

        # --- RViz publisher helper ---
        self.rviz_publisher = RvizPublisherNode(logger=self.get_logger())

        # --- Publish once ---
        self._publish()

    def _publish(self):
        vlm_stack_path = os.path.join(self.output_dir, f'{RUN_NAME}_vlm_object_stack.json')
        if not os.path.exists(vlm_stack_path):
            self.get_logger().error(
                f'[{RUN_NAME}] {RUN_NAME}_vlm_object_stack.json not found at: {vlm_stack_path}'
            )
            return

        with open(vlm_stack_path, 'r') as f:
            vlm_object_stack = json.load(f)

        self.get_logger().info(
            f'[{RUN_NAME}] Loaded {len(vlm_object_stack)} VLM objects from {vlm_stack_path}'
        )

        vlm_markers = self.rviz_publisher.build_vlm_marker_array(
            vlm_object_stack=vlm_object_stack,
            clock=self.get_clock()
        )
        self.vlm_pub.publish(vlm_markers)
        self.get_logger().info(
            f'[{RUN_NAME}] VLM markers published to /vlm_semantic_map_markers — '
            f'{len(vlm_markers.markers)} markers.'
        )

        gt_markers = self.rviz_publisher.build_gt_markers(
            ground_truth=self.ground_truth,
            clock=self.get_clock()
        )
        self.gt_pub.publish(gt_markers)
        self.get_logger().info(
            f'[{RUN_NAME}] GT markers published to /gt_markers — '
            f'{len(gt_markers.markers)} markers.'
        )

        self.get_logger().info(f'[{RUN_NAME}] Done. Keep this node running to hold latched markers in RViz.')


def main(args=None):
    rclpy.init(args=args)
    node = VlmVizNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
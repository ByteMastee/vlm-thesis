import os
import json

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from visualization_msgs.msg import MarkerArray

from semantic_mapping.rviz_publisher_node import RvizPublisherNode


class VlmVizNode(Node):
    def __init__(self):
        super().__init__('vlm_rviz_node')

        # --- Parameters ---
        self.declare_parameter('output_dir',   '/root/UVC_ws/vf_robot_model_ros2/semantic_mapping_output')
        self.declare_parameter('ground_truth', ['chair_1:-3.0:2.0', 'chair_2:-3.5:-2.5', 'couch:3.5:0.0', 'table:2.0:2.5'])

        self.output_dir  = self.get_parameter('output_dir').value
        gt_raw           = self.get_parameter('ground_truth').value

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
        # Load vlm_object_stack.json
        vlm_stack_path = os.path.join(self.output_dir, 'vlm_object_stack2.json')
        if not os.path.exists(vlm_stack_path):
            self.get_logger().error(f'vlm_object_stack.json not found at: {vlm_stack_path}')
            return

        with open(vlm_stack_path, 'r') as f:
            vlm_object_stack = json.load(f)

        self.get_logger().info(f'Loaded {len(vlm_object_stack)} VLM objects from {vlm_stack_path}')

        # Publish VLM markers
        vlm_markers = self.rviz_publisher.build_vlm_marker_array(
            vlm_object_stack=vlm_object_stack,
            clock=self.get_clock()
        )
        self.vlm_pub.publish(vlm_markers)
        self.get_logger().info(
            f'VLM markers published to /vlm_semantic_map_markers — {len(vlm_markers.markers)} markers.'
        )

        # Publish GT markers
        gt_markers = self.rviz_publisher.build_gt_markers(
            ground_truth=self.ground_truth,
            clock=self.get_clock()
        )
        self.gt_pub.publish(gt_markers)
        self.get_logger().info(
            f'GT markers published to /gt_markers — {len(gt_markers.markers)} markers.'
        )

        self.get_logger().info('Done. Keep this node running to hold latched markers in RViz.')


def main(args=None):
    rclpy.init(args=args)
    node = VlmVizNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
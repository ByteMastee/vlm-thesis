import os
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import Odometry
from tf2_msgs.msg import TFMessage
from visualization_msgs.msg import MarkerArray

from semantic_mapping.yolo_map_node import YoloMapNode
from semantic_mapping.rviz_publisher_node import RvizPublisherNode


class RosBridgeNode(Node):
    def __init__(self):
        super().__init__('ros_node')

        # --- Parameters ---
        self.declare_parameter('image_topic',        '/fisheye/front/fisheye_front/image_raw')
        self.declare_parameter('cam_info_topic',     '/fisheye/front/fisheye_front/camera_info')
        self.declare_parameter('odom_topic',         '/odom')
        self.declare_parameter('tf_topic',           '/tf')
        self.declare_parameter('tf_static_topic',    '/tf_static')
        self.declare_parameter('frame_skip',         12)
        self.declare_parameter('confidence',         0.47)
        self.declare_parameter('model_path',         '/root/yolo26m.pt')
        self.declare_parameter('output_dir',         '/root/UVC_ws/vf_robot_model_ros2/semantic_mapping_output')
        self.declare_parameter('min_angle_deg',      8.0)
        self.declare_parameter('dbscan_eps',         1.0)
        self.declare_parameter('dbscan_min_samples', 3)
        self.declare_parameter('ray_length',         10.0)
        self.declare_parameter('process_delay',      110.0)
        self.declare_parameter('ground_truth',       ['chair_1:-3.0:2.0', 'chair_2:-3.5:-2.5', 'couch:3.5:0.0', 'table:2.0:2.5'])

        image_topic     = self.get_parameter('image_topic').value
        cam_info_topic  = self.get_parameter('cam_info_topic').value
        odom_topic      = self.get_parameter('odom_topic').value
        tf_topic        = self.get_parameter('tf_topic').value
        tf_static_topic = self.get_parameter('tf_static_topic').value

        self.frame_skip         = self.get_parameter('frame_skip').value
        self.confidence         = self.get_parameter('confidence').value
        self.model_path         = self.get_parameter('model_path').value
        self.output_dir         = self.get_parameter('output_dir').value
        self.min_angle_deg      = self.get_parameter('min_angle_deg').value
        self.dbscan_eps         = self.get_parameter('dbscan_eps').value
        self.dbscan_min_samples = self.get_parameter('dbscan_min_samples').value
        self.ray_length       = self.get_parameter('ray_length').value
        process_delay         = self.get_parameter('process_delay').value

        gt_raw = self.get_parameter('ground_truth').value
        self.ground_truth = {}
        for entry in gt_raw:
            parts = entry.split(':')
            self.ground_truth[parts[0]] = (float(parts[1]), float(parts[2]))

        os.makedirs(self.output_dir, exist_ok=True)

        # --- State ---
        self.latest_odom        = None
        self.cam_info           = None
        self.is_calibrated      = False
        self.tf_static_msg      = None
        self.frame_count        = 0
        self.processed_count    = 0
        self.process_done       = False
        self.total_start_time   = None
        self.total_compute_time = 0.0
        self.gt_published       = False

        # --- Functional nodes ---
        self.yolo_map_node  = None
        self.rviz_publisher = None

        # --- QoS ---
        tf_static_qos = QoSProfile(
            depth=100,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE
        )
        latched_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )

        # --- Publishers ---
        self.marker_pub      = self.create_publisher(MarkerArray, '/semantic_map_markers', latched_qos)
        self.live_marker_pub = self.create_publisher(MarkerArray, '/semantic_map_live',    10)

        # --- Subscribers ---
        self.create_subscription(CameraInfo, cam_info_topic,  self.cam_info_cb,  10)
        self.create_subscription(Image,      image_topic,     self.image_cb,     10)
        self.create_subscription(Odometry,   odom_topic,      self.odom_cb,      10)
        self.create_subscription(TFMessage,  tf_topic,        self.tf_cb,        10)
        self.create_subscription(TFMessage,  tf_static_topic, self.tf_static_cb, tf_static_qos)

        # --- Process timer ---
        self.create_timer(process_delay, self.process)

        self.get_logger().info(f'ros_node started — process will trigger in {process_delay}s')
        self.get_logger().info('Start bag playback now.')

    # --- Callbacks ---

    def cam_info_cb(self, msg):
        if self.is_calibrated:
            return

        self.cam_info      = msg
        self.is_calibrated = True

        fx = msg.k[0]
        fy = msg.k[4]
        cx = msg.k[2]
        cy = msg.k[5]

        self.get_logger().info(
            f'Camera calibrated — fx:{fx:.4f} fy:{fy:.4f} cx:{cx:.4f} cy:{cy:.4f}'
        )

        self.yolo_map_node = YoloMapNode(
            model_path=self.model_path,
            confidence=self.confidence,
            fx=fx, fy=fy, cx=cx, cy=cy,
            min_angle_deg=self.min_angle_deg,
            dbscan_eps=self.dbscan_eps,
            dbscan_min_samples=self.dbscan_min_samples,
            output_dir=self.output_dir,
            ground_truth=self.ground_truth,
            logger=self.get_logger()
        )

        self.rviz_publisher = RvizPublisherNode(
            logger=self.get_logger()
        )

        if self.tf_static_msg is not None:
            self.yolo_map_node.set_tf_static(self.tf_static_msg)

        # --- Publish GT markers once ---
        if self.ground_truth and not self.gt_published:
            gt_markers = self.rviz_publisher.build_gt_markers(
                self.ground_truth, self.get_clock()
            )
            self.marker_pub.publish(gt_markers)
            self.gt_published = True
            self.get_logger().info('GT markers published.')

        self.get_logger().info('Functional nodes initialized.')

    def image_cb(self, msg):
        self.frame_count += 1

        if not self.is_calibrated:
            return

        if self.latest_odom is None:
            return

        if self.yolo_map_node is None:
            return

        if self.frame_count % self.frame_skip != 0:
            return

        if self.total_start_time is None:
            self.total_start_time = time.time()

        frame_start = time.time()

        rx, ry, frame_rays, frame_candidates = self.yolo_map_node.process_frame(
            msg, self.latest_odom
        )

        frame_elapsed = time.time() - frame_start
        self.processed_count    += 1
        self.total_compute_time += frame_elapsed

        self.get_logger().info(
            f'Frame {self.frame_count} processed | '
            f'count: {self.processed_count} | '
            f'time: {frame_elapsed:.3f}s'
        )

        if rx is None:
            return

        # --- Live RViz publish ---
        all_candidates = self.yolo_map_node.get_all_candidates()

        rays_with_length = [
            (origin, ray, self.ray_length) for origin, ray in frame_rays
        ]

        live_markers = self.rviz_publisher.build_live_markers(
            robot_x=rx,
            robot_y=ry,
            rays=rays_with_length,
            candidates=all_candidates,
            clock=self.get_clock()
        )

        self.live_marker_pub.publish(live_markers)

    def odom_cb(self, msg):
        self.latest_odom = msg

    def tf_cb(self, msg):
        pass

    def tf_static_cb(self, msg):
        self.tf_static_msg = msg
        if self.yolo_map_node is not None:
            self.yolo_map_node.set_tf_static(msg)

    # --- Process ---

    def process(self):
        if self.process_done:
            return

        self.process_done = True

        if not self.is_calibrated:
            self.get_logger().warn('Process triggered — camera not calibrated, aborting.')
            return

        if self.processed_count == 0:
            self.get_logger().warn('Process triggered — no frames processed, aborting.')
            return

        if self.total_start_time is not None:
            total_elapsed = time.time() - self.total_start_time
            self.get_logger().info(
                f'Total wall time: {total_elapsed:.3f}s '
                f'({total_elapsed/60:.2f} min) for {self.processed_count} frames'
            )
            self.get_logger().info(
                f'Pure compute time: {self.total_compute_time:.3f}s | '
                f'avg per frame: {self.total_compute_time/self.processed_count:.4f}s'
            )

        object_stack = self.yolo_map_node.get_object_stack()
        self.get_logger().info(f'Object stack: {list(object_stack.keys())}')

        self.yolo_map_node.save_outputs()

        robot_path_data = {
            'x': self.yolo_map_node.robot_x,
            'y': self.yolo_map_node.robot_y
        }

        marker_array = self.rviz_publisher.build_marker_array(
            object_stack=object_stack,
            ground_truth=self.ground_truth,
            robot_path=robot_path_data,
            clock=self.get_clock()
        )

        self.marker_pub.publish(marker_array)
        self.get_logger().info(
            f'Final markers published to /semantic_map_markers — {len(marker_array.markers)} markers.'
        )

        self.get_logger().info('Processing complete.')


def main(args=None):
    rclpy.init(args=args)
    node = RosBridgeNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
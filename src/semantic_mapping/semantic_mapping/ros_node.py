import os
import time
import json
import threading

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import Odometry
from visualization_msgs.msg import MarkerArray
from std_srvs.srv import Trigger

import tf2_ros

from semantic_mapping.yolo_map_node import YoloMapNode
from semantic_mapping.rviz_publisher_node import RvizPublisherNode

BASE_OUTPUT_DIR = '/root/UVC_ws/vf_robot_model_ros2/Final_Output/Testing'


class RosBridgeNode(Node):
    def __init__(self):
        super().__init__('ros_node')

        # --- Parameters ---
        self.declare_parameter('run_name',           'run_01')
        self.declare_parameter('image_topic',        '/fisheye_front/fisheye_front/image_raw')
        self.declare_parameter('cam_info_topic',     '/fisheye_front/fisheye_front/camera_info')
        self.declare_parameter('odom_topic',         '/odom')
        self.declare_parameter('frame_skip',         12)
        self.declare_parameter('confidence',         0.50)
        self.declare_parameter('model_path',         '/root/yolo26m.pt')
        self.declare_parameter('output_dir',         '')
        self.declare_parameter('min_angle_deg',      8.0)
        self.declare_parameter('dbscan_eps',         1.0)
        self.declare_parameter('dbscan_min_samples', 3)
        self.declare_parameter('ray_length',         8.0)
        self.declare_parameter('process_delay',      95.0)
        self.declare_parameter('env_frame_interval', 20)
        self.declare_parameter('ground_truth',       ['chair_1:-3.0:2.0', 'chair_2:-3.5:-2.5', 'couch:3.5:0.0', 'table:2.0:2.5'])

        # --- NEW: center crop ratio (0.0 = no crop, 0.2 = drop 20% from each side) ---
        self.declare_parameter('crop_margin_ratio',  0.15)

        self.run_name = self.get_parameter('run_name').value

        image_topic    = self.get_parameter('image_topic').value
        cam_info_topic = self.get_parameter('cam_info_topic').value
        odom_topic     = self.get_parameter('odom_topic').value

        self.frame_skip         = self.get_parameter('frame_skip').value
        self.confidence         = self.get_parameter('confidence').value
        self.model_path         = self.get_parameter('model_path').value
        self.min_angle_deg      = self.get_parameter('min_angle_deg').value
        self.dbscan_eps         = self.get_parameter('dbscan_eps').value
        self.dbscan_min_samples = self.get_parameter('dbscan_min_samples').value
        self.ray_length         = self.get_parameter('ray_length').value
        process_delay           = self.get_parameter('process_delay').value
        self.env_frame_interval = self.get_parameter('env_frame_interval').value

        # --- NEW: read and validate crop ratio ---
        self.crop_margin_ratio = float(self.get_parameter('crop_margin_ratio').value)
        if self.crop_margin_ratio < 0.0 or self.crop_margin_ratio >= 0.5:
            self.get_logger().warn(
                f'crop_margin_ratio={self.crop_margin_ratio} invalid (must be in [0.0, 0.5)); '
                f'disabling crop.'
            )
            self.crop_margin_ratio = 0.0

        # Crop window — computed once on first CameraInfo
        self.crop_x0 = 0
        self.crop_y0 = 0
        self.crop_w  = 0
        self.crop_h  = 0

        # --- Output dir: use parameter if provided, else build from run_name ---
        output_dir_param = self.get_parameter('output_dir').value
        if output_dir_param:
            self.output_dir = output_dir_param
        else:
            self.output_dir = os.path.join(BASE_OUTPUT_DIR, self.run_name)

        gt_raw = self.get_parameter('ground_truth').value
        self.ground_truth = {}
        for entry in gt_raw:
            parts = entry.split(':')
            self.ground_truth[parts[0]] = (float(parts[1]), float(parts[2]))

        os.makedirs(self.output_dir, exist_ok=True)

        # --- TF2 Buffer and Listener ---
        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # --- State ---
        self.latest_odom         = None
        self.cam_info            = None
        self.is_calibrated       = False
        self.frame_count         = 0
        self.processed_count     = 0
        self.process_done        = False
        self.total_start_time    = None
        self.total_compute_time  = 0.0
        self.gt_published        = False
        self.cached_marker_array = None
        self.cached_vlm_markers  = None
        self.last_frame_time       = None

        # --- Functional nodes ---
        self.yolo_map_node  = None
        self.rviz_publisher = None

        # --- QoS ---
        latched_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )

        # --- Publishers ---
        self.marker_pub      = self.create_publisher(MarkerArray, '/semantic_map_markers',     latched_qos)
        self.live_marker_pub = self.create_publisher(MarkerArray, '/semantic_map_live',        10)
        self.vlm_marker_pub  = self.create_publisher(MarkerArray, '/vlm_semantic_map_markers', latched_qos)

        # --- Subscribers ---
        self.create_subscription(CameraInfo, cam_info_topic, self.cam_info_cb, 10)
        self.create_subscription(Image,      image_topic,    self.image_cb,    10)
        self.create_subscription(Odometry,   odom_topic,     self.odom_cb,     10)

        # --- Service client: call vlm_label_node to trigger VLM pipeline ---
        self.vlm_client = self.create_client(Trigger, 'run_vlm_pipeline')

        # --- Service server: vlm_label_node calls this when pipeline is done ---
        self.create_service(Trigger, 'vlm_pipeline_done', self._vlm_done_cb)

        # --- Republish timer: keeps markers alive for RViz uncheck/recheck ---
        self.create_timer(3.0, self._republish_markers)

        # --- Process timer ---
        self.create_timer(process_delay, self.process)

        self.get_logger().info(f'ros_node started | RUN_NAME: {self.run_name}')
        self.get_logger().info(f'output_dir: {self.output_dir}')
        self.get_logger().info(f'crop_margin_ratio: {self.crop_margin_ratio}')
        self.get_logger().info(f'process will trigger in {process_delay}s')
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

        # --- NEW: compute center-crop window and adjust principal point ---
        W = int(msg.width)
        H = int(msg.height)

        if self.crop_margin_ratio > 0.0:
            x0 = int(round(W * self.crop_margin_ratio))
            y0 = int(round(H * self.crop_margin_ratio))
            cw = W - 2 * x0
            ch = H - 2 * y0

            self.crop_x0 = x0
            self.crop_y0 = y0
            self.crop_w  = cw
            self.crop_h  = ch

            # Shift principal point — fx, fy are unchanged (no resize, only crop)
            cx_adj = cx - x0
            cy_adj = cy - y0

            self.get_logger().info(
                f'Center crop ENABLED | ratio={self.crop_margin_ratio} | '
                f'original={W}x{H} -> cropped={cw}x{ch} | '
                f'crop_origin=({x0},{y0}) | '
                f'cx: {cx:.2f} -> {cx_adj:.2f} | cy: {cy:.2f} -> {cy_adj:.2f}'
            )
        else:
            self.crop_x0 = 0
            self.crop_y0 = 0
            self.crop_w  = W
            self.crop_h  = H
            cx_adj = cx
            cy_adj = cy
            self.get_logger().info('Center crop DISABLED | using full frame.')

        self.get_logger().info(
            f'Camera calibrated (effective) — fx:{fx:.4f} fy:{fy:.4f} '
            f'cx:{cx_adj:.4f} cy:{cy_adj:.4f}'
        )

        self.yolo_map_node = YoloMapNode(
            model_path=self.model_path,
            confidence=self.confidence,
            fx=fx, fy=fy, cx=cx_adj, cy=cy_adj,
            min_angle_deg=self.min_angle_deg,
            dbscan_eps=self.dbscan_eps,
            dbscan_min_samples=self.dbscan_min_samples,
            output_dir=self.output_dir,
            ground_truth=self.ground_truth,
            logger=self.get_logger(),
            tf_buffer=self.tf_buffer,
            env_frame_interval=self.env_frame_interval,
            run_name=self.run_name
        )

        self.rviz_publisher = RvizPublisherNode(
            logger=self.get_logger()
        )

        # Publish GT markers once
        if self.ground_truth and not self.gt_published:
            gt_markers = self.rviz_publisher.build_gt_markers(
                self.ground_truth, self.get_clock()
            )
            self.marker_pub.publish(gt_markers)
            self.cached_marker_array = gt_markers
            self.gt_published        = True
            self.get_logger().info(f'[{self.run_name}] GT markers published.')

        self.get_logger().info(f'[{self.run_name}] Functional nodes initialized.')

    # --- NEW: crop an incoming sensor_msgs/Image to the configured window ---
    def _crop_image_msg(self, msg):
        """
        Return a new Image message cropped to (crop_x0, crop_y0, crop_w, crop_h).
        fx, fy, cx, cy given to YoloMapNode have already been adjusted for this crop,
        so the triangulation math stays consistent.
        """
        if self.crop_margin_ratio <= 0.0:
            return msg

        enc = msg.encoding
        if enc in ('rgb8', 'bgr8'):
            channels = 3
        elif enc == 'mono8':
            channels = 1
        else:
            # Unsupported — pass through; yolo_map_node._decode_image will warn
            return msg

        W = int(msg.width)
        H = int(msg.height)

        # Defensive: if bag resolution changed vs first CameraInfo, skip crop
        if W != (self.crop_w + 2 * self.crop_x0) or H != (self.crop_h + 2 * self.crop_y0):
            return msg

        arr = np.frombuffer(bytes(msg.data), dtype=np.uint8)
        if channels == 1:
            arr = arr.reshape((H, W))
            crop = arr[self.crop_y0:self.crop_y0 + self.crop_h,
                       self.crop_x0:self.crop_x0 + self.crop_w]
        else:
            arr = arr.reshape((H, W, channels))
            crop = arr[self.crop_y0:self.crop_y0 + self.crop_h,
                       self.crop_x0:self.crop_x0 + self.crop_w, :]

        # Contiguous buffer for tobytes()
        crop = np.ascontiguousarray(crop)

        new_msg          = Image()
        new_msg.header   = msg.header
        new_msg.height   = self.crop_h
        new_msg.width    = self.crop_w
        new_msg.encoding = enc
        new_msg.is_bigendian = msg.is_bigendian
        new_msg.step     = self.crop_w * channels
        new_msg.data     = crop.tobytes()
        return new_msg

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

        self.last_frame_time = time.time()
        frame_start = time.time()

        # --- NEW: apply center crop before handing off to yolo_map_node ---
        msg_for_yolo = self._crop_image_msg(msg)

        rx, ry, frame_rays, frame_candidates = self.yolo_map_node.process_frame(
            msg_for_yolo, self.latest_odom
        )

        frame_elapsed = time.time() - frame_start
        self.processed_count    += 1
        self.total_compute_time += frame_elapsed

        self.get_logger().info(
            f'[{self.run_name}] Frame {self.frame_count} processed | '
            f'count: {self.processed_count} | '
            f'time: {frame_elapsed:.3f}s'
        )

        if rx is None:
            return

        rays_with_length = [
            (origin_2d, ray_2d, self.ray_length)
            for origin_2d, ray_2d in frame_rays
        ]

        live_markers = self.rviz_publisher.build_live_markers(
            robot_x=rx,
            robot_y=ry,
            rays=rays_with_length,
            candidates=self.yolo_map_node.get_all_candidates(),
            clock=self.get_clock()
        )

        self.live_marker_pub.publish(live_markers)

    def odom_cb(self, msg):
        self.latest_odom = msg

    # --- Process ---

    def process(self):
        if self.process_done:
            return

        if self.last_frame_time is None:
            return

        if time.time() - self.last_frame_time < 5.0:
            return

        self.process_done = True

        if not self.is_calibrated:
            self.get_logger().warn(f'[{self.run_name}] Process triggered — camera not calibrated, aborting.')
            return

        if self.processed_count == 0:
            self.get_logger().warn(f'[{self.run_name}] Process triggered — no frames processed, aborting.')
            return

        if self.total_start_time is not None:
            total_elapsed = time.time() - self.total_start_time
            self.get_logger().info(
                f'[{self.run_name}] Total wall time: {total_elapsed:.3f}s '
                f'({total_elapsed/60:.2f} min) for {self.processed_count} frames'
            )
            self.get_logger().info(
                f'[{self.run_name}] Pure compute time: {self.total_compute_time:.3f}s | '
                f'avg per frame: {self.total_compute_time/self.processed_count:.4f}s'
            )

        object_stack = self.yolo_map_node.get_object_stack()
        self.get_logger().info(f'[{self.run_name}] Object stack: {list(object_stack.keys())}')

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
        self.cached_marker_array = marker_array
        self.get_logger().info(
            f'[{self.run_name}] Final markers published to /semantic_map_markers — '
            f'{len(marker_array.markers)} markers.'
        )

        self.get_logger().info(f'[{self.run_name}] YOLO mapping complete — calling VLM pipeline...')

        # --- Call VLM pipeline in separate thread ---
        thread = threading.Thread(target=self._call_vlm_service, daemon=True)
        thread.start()

    # --- VLM service call (runs in separate thread) ---

    def _call_vlm_service(self):
        if not self.vlm_client.wait_for_service(timeout_sec=10.0):
            self.get_logger().warn(
                f'[{self.run_name}] run_vlm_pipeline service not available — VLM pipeline skipped.'
            )
            return

        request = Trigger.Request()
        future  = self.vlm_client.call_async(request)

        timeout = 30.0
        start   = time.time()
        while not future.done():
            if time.time() - start > timeout:
                self.get_logger().warn(f'[{self.run_name}] VLM service call timed out.')
                return
            time.sleep(0.1)

        try:
            result = future.result()
            if result.success:
                self.get_logger().info(
                    f'[{self.run_name}] VLM pipeline triggered successfully: {result.message}'
                )
            else:
                self.get_logger().warn(
                    f'[{self.run_name}] VLM pipeline trigger failed: {result.message}'
                )
        except Exception as e:
            self.get_logger().error(f'[{self.run_name}] VLM service call error: {e}')

    # --- VLM pipeline done callback (called by vlm_label_node) ---

    def _vlm_done_cb(self, request, response):
        self.get_logger().info(f'[{self.run_name}] VLM pipeline complete — publishing VLM markers.')

        vlm_stack_path = os.path.join(self.output_dir, f'{self.run_name}_vlm_object_stack.json')
        if not os.path.exists(vlm_stack_path):
            self.get_logger().error(
                f'[{self.run_name}] {self.run_name}_vlm_object_stack.json not found — cannot publish VLM markers.'
            )
            response.success = False
            response.message = 'vlm_object_stack.json not found.'
            return response

        with open(vlm_stack_path, 'r') as f:
            vlm_object_stack = json.load(f)

        vlm_markers = self.rviz_publisher.build_vlm_marker_array(
            vlm_object_stack=vlm_object_stack,
            clock=self.get_clock()
        )

        self.vlm_marker_pub.publish(vlm_markers)
        self.cached_vlm_markers = vlm_markers
        self.get_logger().info(
            f'[{self.run_name}] VLM markers published to /vlm_semantic_map_markers — '
            f'{len(vlm_markers.markers)} markers.'
        )

        response.success = True
        response.message = 'VLM markers published.'
        return response

    # --- Republish timer ---

    def _republish_markers(self):
        if self.cached_marker_array is not None:
            self.marker_pub.publish(self.cached_marker_array)
        if self.cached_vlm_markers is not None:
            self.vlm_marker_pub.publish(self.cached_vlm_markers)


def main(args=None):
    rclpy.init(args=args)
    node = RosBridgeNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
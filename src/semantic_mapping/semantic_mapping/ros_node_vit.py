import os
import time
import json
import threading
import gc

import torch
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseWithCovarianceStamped
from visualization_msgs.msg import MarkerArray
from std_srvs.srv import Trigger

import tf2_ros

from semantic_mapping.sam2_map_node import SAM2MapNode
from semantic_mapping.rviz_publisher_node import RvizPublisherNode

BASE_OUTPUT_DIR = '/root/UVC_ws/vf_robot_model_ros2/Final_Output/Testing'


class RosBridgeNodeVit(Node):
    def __init__(self):
        super().__init__('ros_node_vit')

        self.declare_parameter('run_name',               'run_01')
        self.declare_parameter('image_topic',            '/fisheye_front/image_rect')
        self.declare_parameter('cam_info_topic',         '/fisheye_front/camera_info')
        self.declare_parameter('odom_topic',             '/localization_pose')
        self.declare_parameter('frame_skip',             10)
        self.declare_parameter('frame_interval_sec',     0.0)
        self.declare_parameter('output_dir',             '')
        self.declare_parameter('min_angle_deg',          8.0)
        self.declare_parameter('dbscan_eps',             1.0)
        self.declare_parameter('dbscan_min_samples',     3)
        self.declare_parameter('ray_length',             8.0)
        self.declare_parameter('env_frame_interval',     20)
        self.declare_parameter('min_candidates',         3)
        self.declare_parameter('ground_truth',           [''])
        self.declare_parameter('sam2_checkpoint',        '/root/sam2_checkpoints/sam2.1_hiera_small.pt')
        self.declare_parameter('sam2_model_cfg',         'configs/sam2.1/sam2.1_hiera_s.yaml')
        self.declare_parameter('points_per_side',        16)
        self.declare_parameter('pred_iou_thresh',        0.82)
        self.declare_parameter('stability_score_thresh', 0.85)
        self.declare_parameter('min_mask_region_area',   1500)
        self.declare_parameter('max_mask_area_fraction', 0.20)
        self.declare_parameter('max_regions',            12)

        self.run_name = self.get_parameter('run_name').value

        image_topic    = self.get_parameter('image_topic').value
        cam_info_topic = self.get_parameter('cam_info_topic').value
        odom_topic     = self.get_parameter('odom_topic').value

        self.frame_skip             = self.get_parameter('frame_skip').value
        self.frame_interval_sec     = self.get_parameter('frame_interval_sec').value
        self.min_angle_deg          = self.get_parameter('min_angle_deg').value
        self.dbscan_eps             = self.get_parameter('dbscan_eps').value
        self.dbscan_min_samples     = self.get_parameter('dbscan_min_samples').value
        self.ray_length             = self.get_parameter('ray_length').value
        self.env_frame_interval     = self.get_parameter('env_frame_interval').value
        self.min_candidates         = self.get_parameter('min_candidates').value
        self.sam2_checkpoint        = self.get_parameter('sam2_checkpoint').value
        self.sam2_model_cfg         = self.get_parameter('sam2_model_cfg').value
        self.points_per_side        = self.get_parameter('points_per_side').value
        self.pred_iou_thresh        = self.get_parameter('pred_iou_thresh').value
        self.stability_score_thresh = self.get_parameter('stability_score_thresh').value
        self.min_mask_region_area   = self.get_parameter('min_mask_region_area').value
        self.max_mask_area_fraction = self.get_parameter('max_mask_area_fraction').value
        self.max_regions            = self.get_parameter('max_regions').value

        output_dir_param = self.get_parameter('output_dir').value
        if output_dir_param:
            self.output_dir = output_dir_param
        else:
            self.output_dir = os.path.join(BASE_OUTPUT_DIR, self.run_name)

        gt_raw = self.get_parameter('ground_truth').value
        self.ground_truth = {}
        if gt_raw and gt_raw != ['']:
            for entry in gt_raw:
                parts = entry.split(':')
                if len(parts) == 3:
                    self.ground_truth[parts[0]] = (float(parts[1]), float(parts[2]))

        os.makedirs(self.output_dir, exist_ok=True)

        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

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
        self.last_processed_time = 0.0

        self.sam2_map_node  = None
        self.rviz_publisher = None

        latched_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)

        self.marker_pub      = self.create_publisher(MarkerArray, '/vit_semantic_map_markers',     latched_qos)
        self.live_marker_pub = self.create_publisher(MarkerArray, '/vit_semantic_map_live',        10)
        self.vlm_marker_pub  = self.create_publisher(MarkerArray, '/vit_vlm_semantic_map_markers', latched_qos)

        self.create_subscription(CameraInfo,                  cam_info_topic, self.cam_info_cb, 10)
        self.create_subscription(Image,                       image_topic,    self.image_cb,    10)
        self.create_subscription(PoseWithCovarianceStamped,   odom_topic,     self.odom_cb,     10)

        self.vlm_client = self.create_client(Trigger, 'run_vlm_pipeline_vit')
        self.create_service(Trigger, 'vlm_pipeline_done_vit', self._vlm_done_cb)
        self.create_service(Trigger, 'finalize_map_vit',      self._finalize_map_cb)

        self.create_timer(3.0, self._republish_markers)

        self.get_logger().info(f'ros_node_vit started | RUN_NAME: {self.run_name}')
        self.get_logger().info(f'output_dir: {self.output_dir}')
        self.get_logger().info(f'ground_truth: {"disabled" if not self.ground_truth else self.ground_truth}')
        if self.frame_interval_sec > 0.0:
            self.get_logger().info(f'Frame mode: time-based — 1 frame per {self.frame_interval_sec}s')
        else:
            self.get_logger().info(f'Frame mode: skip-based — 1 frame per {self.frame_skip} frames')
        self.get_logger().info('Live stream mode — call /finalize_map_vit service to finalize.')

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

        self.sam2_map_node = SAM2MapNode(
            checkpoint_path=self.sam2_checkpoint,
            model_cfg=self.sam2_model_cfg,
            fx=fx, fy=fy, cx=cx, cy=cy,
            min_angle_deg=self.min_angle_deg,
            dbscan_eps=self.dbscan_eps,
            dbscan_min_samples=self.dbscan_min_samples,
            output_dir=self.output_dir,
            ground_truth=self.ground_truth,
            logger=self.get_logger(),
            tf_buffer=self.tf_buffer,
            env_frame_interval=self.env_frame_interval,
            run_name=self.run_name,
            min_candidates=self.min_candidates,
            points_per_side=self.points_per_side,
            pred_iou_thresh=self.pred_iou_thresh,
            stability_score_thresh=self.stability_score_thresh,
            min_mask_region_area=self.min_mask_region_area,
            max_mask_area_fraction=self.max_mask_area_fraction,
            max_regions=self.max_regions
        )

        self.rviz_publisher = RvizPublisherNode(logger=self.get_logger())

        if self.ground_truth and not self.gt_published:
            gt_markers = self.rviz_publisher.build_gt_markers(
                self.ground_truth, self.get_clock()
            )
            self.marker_pub.publish(gt_markers)
            self.cached_marker_array = gt_markers
            self.gt_published        = True
            self.get_logger().info(f'[{self.run_name}] GT markers published.')
        else:
            self.get_logger().info(f'[{self.run_name}] No GT — skipping GT markers.')

        self.get_logger().info(f'[{self.run_name}] SAM2 node initialized.')

    def image_cb(self, msg):
        self.frame_count += 1

        if not self.is_calibrated:
            return
        if self.latest_odom is None:
            return
        if self.sam2_map_node is None:
            return
        if self.process_done:
            return

        now = time.time()

        if self.frame_interval_sec > 0.0:
            if now - self.last_processed_time < self.frame_interval_sec:
                return
        else:
            if self.frame_count % self.frame_skip != 0:
                return

        self.last_processed_time = now

        if self.total_start_time is None:
            self.total_start_time = now

        frame_start = time.time()

        rx, ry, frame_rays, frame_candidates = self.sam2_map_node.process_frame(
            msg, self.latest_odom
        )

        frame_elapsed = time.time() - frame_start
        self.processed_count    += 1
        self.total_compute_time += frame_elapsed

        self.get_logger().info(
            f'[{self.run_name}] Frame {self.frame_count} | '
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
            candidates=self.sam2_map_node.get_all_candidates(),
            clock=self.get_clock()
        )
        self.live_marker_pub.publish(live_markers)

    def odom_cb(self, msg):
        self.latest_odom = msg

    def _unload_sam2(self):
        if self.sam2_map_node is not None:
            self.get_logger().info(f'[{self.run_name}] Unloading SAM2 from GPU...')
            del self.sam2_map_node.mask_generator
            del self.sam2_map_node
            self.sam2_map_node = None
            gc.collect()
            torch.cuda.empty_cache()
            self.get_logger().info(f'[{self.run_name}] SAM2 unloaded. VRAM freed for VLM.')

    def _finalize_map_cb(self, request, response):
        if self.process_done:
            response.success = False
            response.message = 'Map already finalized.'
            return response

        if not self.is_calibrated:
            response.success = False
            response.message = 'Camera not calibrated yet.'
            return response

        if self.processed_count == 0:
            response.success = False
            response.message = 'No frames processed yet.'
            return response

        self.process_done = True

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

        object_stack = self.sam2_map_node.get_object_stack()
        self.get_logger().info(f'[{self.run_name}] Object stack: {list(object_stack.keys())}')

        self.sam2_map_node.save_outputs()

        robot_path_data = {
            'x': self.sam2_map_node.robot_x,
            'y': self.sam2_map_node.robot_y
        }

        marker_array = self.rviz_publisher.build_marker_array(
            object_stack=object_stack,
            ground_truth=self.ground_truth,
            robot_path=robot_path_data,
            clock=self.get_clock()
        )

        self.marker_pub.publish(marker_array)
        self.cached_marker_array = marker_array
        self.get_logger().info(f'[{self.run_name}] VIT markers published.')

        self._unload_sam2()

        thread = threading.Thread(target=self._call_vlm_service, daemon=True)
        thread.start()

        response.success = True
        response.message = f'Map finalized — {len(object_stack)} objects detected.'
        return response

    def _call_vlm_service(self):
        if not self.vlm_client.wait_for_service(timeout_sec=10.0):
            self.get_logger().warn(f'[{self.run_name}] run_vlm_pipeline_vit not available.')
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
                self.get_logger().info(f'[{self.run_name}] VLM VIT pipeline triggered.')
            else:
                self.get_logger().warn(f'[{self.run_name}] VLM trigger failed: {result.message}')
        except Exception as e:
            self.get_logger().error(f'[{self.run_name}] VLM service error: {e}')

    def _vlm_done_cb(self, request, response):
        self.get_logger().info(f'[{self.run_name}] VLM VIT pipeline complete — publishing markers.')

        vlm_stack_path = os.path.join(self.output_dir, f'{self.run_name}_vit_vlm_object_stack.json')
        if not os.path.exists(vlm_stack_path):
            self.get_logger().error(f'[{self.run_name}] vit_vlm_object_stack.json not found.')
            response.success = False
            response.message = 'vit_vlm_object_stack.json not found.'
            return response

        with open(vlm_stack_path, 'r') as f:
            vlm_object_stack = json.load(f)

        robot_path_json = os.path.join(self.output_dir, f'{self.run_name}_vit_robot_path.json')
        if os.path.exists(robot_path_json):
            with open(robot_path_json, 'r') as f:
                robot_path_data = json.load(f)
        else:
            robot_path_data = None

        filtered_marker_array = self.rviz_publisher.build_marker_array(
            object_stack=vlm_object_stack,
            ground_truth=self.ground_truth,
            robot_path=robot_path_data,
            clock=self.get_clock()
        )
        self.marker_pub.publish(filtered_marker_array)
        self.cached_marker_array = filtered_marker_array

        vlm_markers = self.rviz_publisher.build_vlm_marker_array(
            vlm_object_stack=vlm_object_stack,
            clock=self.get_clock()
        )
        self.vlm_marker_pub.publish(vlm_markers)
        self.cached_vlm_markers = vlm_markers
        self.get_logger().info(
            f'[{self.run_name}] VIT VLM markers published — {len(vlm_markers.markers)} markers.'
        )

        response.success = True
        response.message = 'VIT VLM markers published.'
        return response

    def _republish_markers(self):
        if self.cached_marker_array is not None:
            self.marker_pub.publish(self.cached_marker_array)
        if self.cached_vlm_markers is not None:
            self.vlm_marker_pub.publish(self.cached_vlm_markers)


def main(args=None):
    rclpy.init(args=args)
    node = RosBridgeNodeVit()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
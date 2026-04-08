import os
import sys
import json
import time
import cv2
import numpy as np
from sklearn.cluster import DBSCAN

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
from depth_anything_3.api import DepthAnything3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import Odometry
from visualization_msgs.msg import MarkerArray

import tf2_ros

from ultralytics import YOLO

from semantic_mapping.rviz_publisher_node import RvizPublisherNode


class DA3MapNode(Node):
    def __init__(self):
        super().__init__('da3_map_node')

        # --- Parameters ---
        self.declare_parameter('image_topic',        '/fisheye_front/fisheye_front/image_raw')
        self.declare_parameter('cam_info_topic',     '/fisheye_front/fisheye_front/camera_info')
        self.declare_parameter('odom_topic',         '/odom')
        self.declare_parameter('frame_skip',         13)
        self.declare_parameter('yolo_confidence',    0.65)
        self.declare_parameter('yolo_model_path',    '/root/yolo26m.pt')
        self.declare_parameter('da3_model_path',     '/root/.cache/huggingface/hub/models--depth-anything--DA3METRIC-LARGE/snapshots/4010e39f3634a45bc60553321fb49fb760bd594e')
        self.declare_parameter('output_dir',         '/root/UVC_ws/vf_robot_model_ros2/semantic_mapping_output')
        self.declare_parameter('dbscan_eps',         1.0)
        self.declare_parameter('dbscan_min_samples', 3)
        self.declare_parameter('process_delay',      95.0)
        self.declare_parameter('ground_truth',       ['chair_1:-3.0:2.0', 'chair_2:-3.5:-2.5', 'couch:3.5:0.0', 'table:2.0:2.5'])

        image_topic    = self.get_parameter('image_topic').value
        cam_info_topic = self.get_parameter('cam_info_topic').value
        odom_topic     = self.get_parameter('odom_topic').value

        self.frame_skip         = self.get_parameter('frame_skip').value
        self.yolo_confidence    = self.get_parameter('yolo_confidence').value
        self.yolo_model_path    = self.get_parameter('yolo_model_path').value
        self.da3_model_path     = self.get_parameter('da3_model_path').value
        self.output_dir         = self.get_parameter('output_dir').value
        self.dbscan_eps         = self.get_parameter('dbscan_eps').value
        self.dbscan_min_samples = self.get_parameter('dbscan_min_samples').value
        process_delay           = self.get_parameter('process_delay').value

        gt_raw = self.get_parameter('ground_truth').value
        self.ground_truth = {}
        for entry in gt_raw:
            parts = entry.split(':')
            self.ground_truth[parts[0]] = (float(parts[1]), float(parts[2]))

        os.makedirs(self.output_dir, exist_ok=True)

        # --- TF2 ---
        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # --- State ---
        self.latest_odom        = None
        self.is_calibrated      = False
        self.fx = self.fy = self.cx = self.cy = None
        self.frame_count        = 0
        self.processed_count    = 0
        self.process_done       = False
        self.total_start_time   = None
        self.total_compute_time = 0.0
        self.gt_published       = False

        # Per-label candidate accumulator: {label: [(x, y), ...]}
        self.candidate_stack = {}
        self.object_stack    = {}
        self.robot_x         = []
        self.robot_y         = []

        # --- Models (loaded after cam_info arrives) ---
        self.yolo_model = None
        self.da3_model  = None
        self.device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # --- RViz publisher (reuse existing node) ---
        self.rviz_publisher = RvizPublisherNode(logger=self.get_logger())

        # --- QoS ---
        latched_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )

        # --- Publishers ---
        self.marker_pub = self.create_publisher(MarkerArray, '/da3_semantic_map_markers', latched_qos)

        # --- Subscribers ---
        self.create_subscription(CameraInfo, cam_info_topic, self.cam_info_cb, 10)
        self.create_subscription(Image,      image_topic,    self.image_cb,    10)
        self.create_subscription(Odometry,   odom_topic,     self.odom_cb,     10)

        # --- Process timer ---
        self.create_timer(process_delay, self.process)

        self.get_logger().info(f'da3_map_node started — process will trigger in {process_delay}s')
        self.get_logger().info('Start bag playback now.')

    # --- Callbacks ---

    def cam_info_cb(self, msg):
        if self.is_calibrated:
            return

        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]
        self.is_calibrated = True

        self.get_logger().info(
            f'Camera calibrated — fx:{self.fx:.4f} fy:{self.fy:.4f} '
            f'cx:{self.cx:.4f} cy:{self.cy:.4f}'
        )

        # Load YOLO
        self.get_logger().info(f'Loading YOLO: {self.yolo_model_path}')
        self.yolo_model = YOLO(self.yolo_model_path)
        self.get_logger().info('YOLO loaded.')

        # Load DA3Metric-Large from local directory.
        # from_pretrained accepts pathlib.Path for local directories directly.
        self.get_logger().info(f'Loading DA3Metric-Large: {self.da3_model_path}')
        from pathlib import Path as _Path
        self.da3_model = DepthAnything3.from_pretrained(_Path(self.da3_model_path))
        self.da3_model = self.da3_model.to(self.device)
        self.da3_model.eval()
        self.get_logger().info('DA3Metric-Large loaded.')

        # Publish GT markers once
        if self.ground_truth and not self.gt_published:
            gt_markers = self.rviz_publisher.build_gt_markers(
                self.ground_truth, self.get_clock()
            )
            self.marker_pub.publish(gt_markers)
            self.gt_published = True
            self.get_logger().info('GT markers published.')

    def image_cb(self, msg):
        self.frame_count += 1

        if not self.is_calibrated:
            return
        if self.latest_odom is None:
            return
        if self.yolo_model is None or self.da3_model is None:
            return
        if self.frame_count % self.frame_skip != 0:
            return

        if self.total_start_time is None:
            self.total_start_time = time.time()

        frame_start = time.time()
        self._process_frame(msg, self.latest_odom)
        frame_elapsed = time.time() - frame_start

        self.processed_count    += 1
        self.total_compute_time += frame_elapsed

        self.get_logger().info(
            f'Frame {self.frame_count} processed | '
            f'count: {self.processed_count} | '
            f'time: {frame_elapsed:.3f}s'
        )

    def odom_cb(self, msg):
        self.latest_odom = msg

    # --- Core frame processing ---

    def _process_frame(self, image_msg, odom_msg):
        img = self._decode_image(image_msg)
        if img is None:
            return

        rx, ry, _ = self._extract_odom(odom_msg)
        self.robot_x.append(rx)
        self.robot_y.append(ry)

        # Step 1: Run DA3Metric — get depth map [H, W] in net_output units
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        with torch.no_grad():
            prediction = self.da3_model.inference([img_rgb])
        net_output = prediction.depth[0]  # [H, W] numpy float32

        # Step 2: Convert net_output to metric depth in metres
        # Formula: metric_depth = focal * net_output / 300.
        focal = (self.fx + self.fy) / 2.0
        depth_map = focal * net_output / 300.0  # [H, W] metres

        # Step 3: Run YOLO detections
        results = self.yolo_model(img, conf=self.yolo_confidence, verbose=False)

        # Step 4: Look up TF once per frame — optical_frame -> odom
        try:
            tf_stamped = self.tf_buffer.lookup_transform(
                'odom',
                'camera_fisheye_front_optical_frame',
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1)
            )
        except Exception as e:
            self.get_logger().warn(f'TF lookup failed: {e}')
            return

        R = self._quat_to_rotation_matrix(tf_stamped.transform.rotation)
        t = np.array([
            tf_stamped.transform.translation.x,
            tf_stamped.transform.translation.y,
            tf_stamped.transform.translation.z
        ])

        # DA3 resizes the image internally before inference.
        # depth_map resolution differs from original image resolution.
        # Compute scale factors to map YOLO pixel coords -> depth map coords.
        img_h, img_w = img.shape[:2]
        dh, dw = depth_map.shape
        scale_x = dw / img_w
        scale_y = dh / img_h

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                label  = self.yolo_model.names[cls_id]

                # Centroid pixel in original image space
                px_img = (x1 + x2) // 2
                py_img = (y1 + y2) // 2

                # Scale centroid and box to depth map resolution
                px_d = int(np.clip(px_img * scale_x, 0, dw - 1))
                py_d = int(np.clip(py_img * scale_y, 0, dh - 1))

                # Scale bounding box to depth map resolution for patch crop
                x1_d = int(np.clip(x1 * scale_x, 0, dw - 1))
                x2_d = int(np.clip(x2 * scale_x, 0, dw - 1))
                y1_d = int(np.clip(y1 * scale_y, 0, dh - 1))
                y2_d = int(np.clip(y2 * scale_y, 0, dh - 1))

                # Sample median depth over 30% center crop in depth map space
                box_w_d = x2_d - x1_d
                box_h_d = y2_d - y1_d
                crop_w = max(1, int(box_w_d * 0.3))
                crop_h = max(1, int(box_h_d * 0.3))
                crop_x1 = int(np.clip(px_d - crop_w // 2, 0, dw - 1))
                crop_x2 = int(np.clip(px_d + crop_w // 2, 0, dw - 1))
                crop_y1 = int(np.clip(py_d - crop_h // 2, 0, dh - 1))
                crop_y2 = int(np.clip(py_d + crop_h // 2, 0, dh - 1))
                depth_patch = depth_map[crop_y1:crop_y2 + 1, crop_x1:crop_x2 + 1]

                if depth_patch.size == 0:
                    continue

                z = float(np.median(depth_patch))

                if z <= 0.1 or z > 20.0:
                    continue

                # Step 5: Back-project original image centroid pixel + depth to 3D
                # Use original image pixel coords with original camera intrinsics
                x_cam = (px_img - self.cx) * z / self.fx
                y_cam = (py_img - self.cy) * z / self.fy
                z_cam = z
                point_optical = np.array([x_cam, y_cam, z_cam])

                # Step 6: Transform point to odom frame
                point_odom = R @ point_optical + t

                # Step 7: Store (x, y) candidate
                obj_x = float(point_odom[0])
                obj_y = float(point_odom[1])

                if label not in self.candidate_stack:
                    self.candidate_stack[label] = []

                self.candidate_stack[label].append((obj_x, obj_y))

    # --- Triggered once after bag ends ---

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

        # Cluster candidates
        self.object_stack = {}
        for label, candidates in self.candidate_stack.items():
            entries = self._cluster_candidates(candidates, label)
            self.object_stack.update(entries)

        self.get_logger().info(f'Object stack: {list(self.object_stack.keys())}')

        # Save outputs
        self._save_outputs()

        # Publish final markers
        robot_path_data = {'x': self.robot_x, 'y': self.robot_y}
        marker_array = self.rviz_publisher.build_marker_array(
            object_stack=self.object_stack,
            ground_truth=self.ground_truth,
            robot_path=robot_path_data,
            clock=self.get_clock()
        )
        self.marker_pub.publish(marker_array)
        self.get_logger().info(
            f'Final markers published to /da3_semantic_map_markers — '
            f'{len(marker_array.markers)} markers.'
        )
        self.get_logger().info('Processing complete.')

    # --- Save outputs ---

    def _save_outputs(self):
        json_path = os.path.join(self.output_dir, 'DA3_object3.json')
        with open(json_path, 'w') as f:
            json.dump(self.object_stack, f, indent=2)
        self.get_logger().info(f'Object stack saved: {json_path}')

        robot_path_json = os.path.join(self.output_dir, 'DA3_path1.json')
        with open(robot_path_json, 'w') as f:
            json.dump({'x': self.robot_x, 'y': self.robot_y}, f)
        self.get_logger().info(f'Robot path saved: {robot_path_json}')

        plot_path = self._save_map_plot()
        self.get_logger().info(f'Map plot saved: {plot_path}')

    def _save_map_plot(self):
        plt.figure(figsize=(10, 10))

        if self.robot_x and self.robot_y:
            plt.plot(self.robot_x, self.robot_y, 'b-', linewidth=1.0, alpha=0.5)
            plt.plot(self.robot_x[0],  self.robot_y[0],  'go', markersize=8)
            plt.plot(self.robot_x[-1], self.robot_y[-1], 'rs', markersize=8)

        for label, (gx, gy) in self.ground_truth.items():
            plt.plot(gx, gy, 'g^', markersize=12)
            plt.annotate(f'GT: {label}\n({gx},{gy})', (gx, gy),
                         textcoords='offset points', xytext=(8, 8),
                         fontsize=9, color='green')

        colors = ['red', 'orange', 'purple', 'cyan', 'magenta']
        for i, (label, data) in enumerate(self.object_stack.items()):
            ox    = data['x']
            oy    = data['y']
            color = colors[i % len(colors)]
            plt.plot(ox, oy, '*', markersize=15, color=color)
            plt.annotate(f'Det: {label}\n({ox:.2f},{oy:.2f})', (ox, oy),
                         textcoords='offset points', xytext=(8, -18),
                         fontsize=9, color=color)

            best_dist = float('inf')
            best_gx, best_gy = None, None
            for gt_label, (gx, gy) in self.ground_truth.items():
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
        plt.title('DA3 Metric Semantic Map — Detected vs Ground Truth')
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

        plot_path = os.path.join(self.output_dir, 'DA3_map3.png')
        plt.savefig(plot_path, dpi=150)
        plt.close()
        return plot_path

    # --- Helpers ---

    def _cluster_candidates(self, candidates, label):
        object_entries = {}
        if len(candidates) == 0:
            return object_entries

        pts = np.array(candidates)

        if len(pts) < self.dbscan_min_samples:
            object_entries[label] = {
                'x': round(float(np.median(pts[:, 0])), 4),
                'y': round(float(np.median(pts[:, 1])), 4),
                'num_candidates': len(pts)
            }
            return object_entries

        db        = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples).fit(pts)
        labels_db = db.labels_

        unique_clusters = set(labels_db)
        unique_clusters.discard(-1)

        if len(unique_clusters) == 0:
            return object_entries

        for cluster_id in sorted(unique_clusters):
            cluster_pts    = pts[labels_db == cluster_id]
            instance_label = label if len(unique_clusters) == 1 else f'{label}_{cluster_id + 1}'
            object_entries[instance_label] = {
                'x': round(float(np.median(cluster_pts[:, 0])), 4),
                'y': round(float(np.median(cluster_pts[:, 1])), 4),
                'num_candidates': len(cluster_pts)
            }

        return object_entries

    def _decode_image(self, msg):
        img_array = np.frombuffer(bytes(msg.data), dtype=np.uint8)
        if msg.encoding == 'rgb8':
            img = img_array.reshape((msg.height, msg.width, 3))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif msg.encoding == 'bgr8':
            img = img_array.reshape((msg.height, msg.width, 3))
        elif msg.encoding == 'mono8':
            img = img_array.reshape((msg.height, msg.width))
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            self.get_logger().warn(f'Unsupported encoding: {msg.encoding}')
            return None
        return img

    def _extract_odom(self, msg):
        rx  = msg.pose.pose.position.x
        ry  = msg.pose.pose.position.y
        q   = msg.pose.pose.orientation
        yaw = np.arctan2(
            2*(q.w*q.z + q.x*q.y),
            1 - 2*(q.y**2 + q.z**2)
        )
        return rx, ry, yaw

    def _quat_to_rotation_matrix(self, q):
        qx, qy, qz, qw = q.x, q.y, q.z, q.w
        return np.array([
            [1 - 2*(qy**2 + qz**2),  2*(qx*qy - qz*qw),  2*(qx*qz + qy*qw)],
            [2*(qx*qy + qz*qw),  1 - 2*(qx**2 + qz**2),  2*(qy*qz - qx*qw)],
            [2*(qx*qz - qy*qw),  2*(qy*qz + qx*qw),  1 - 2*(qx**2 + qy**2)]
        ])


def main(args=None):
    rclpy.init(args=args)
    node = DA3MapNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
import os
import json
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='scipy')

import cv2
import numpy as np
from sklearn.cluster import DBSCAN

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ultralytics import YOLO

import tf2_ros
import rclpy

# --- Run name: must match ros_node.py ---
RUN_NAME = 'run_01'


class YoloMapNode:
    def __init__(
        self,
        model_path,
        confidence,
        fx, fy, cx, cy,
        min_angle_deg,
        dbscan_eps,
        dbscan_min_samples,
        output_dir,
        ground_truth,
        logger,
        tf_buffer,
        env_frame_interval=20
    ):
        self.confidence         = confidence
        self.fx                 = fx
        self.fy                 = fy
        self.cx                 = cx
        self.cy                 = cy
        self.min_angle_deg      = min_angle_deg
        self.dbscan_eps         = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.output_dir         = output_dir
        self.ground_truth       = ground_truth
        self.logger             = logger
        self.tf_buffer          = tf_buffer
        self.env_frame_interval = env_frame_interval

        self.ray_stack         = {}
        self.candidate_stack   = {}
        self.object_stack      = {}

        self.robot_x = []
        self.robot_y = []

        self.total_triangulated    = 0
        self.total_skipped         = 0
        self.processed_frame_count = 0

        # --- Output folders ---
        self.det_objects_dir = os.path.join(output_dir, 'detections', 'objects')
        self.env_frames_dir  = os.path.join(output_dir, 'env_frames')
        os.makedirs(self.det_objects_dir, exist_ok=True)
        os.makedirs(self.env_frames_dir,  exist_ok=True)

        self.logger.info(f'Loading YOLO model: {model_path}')
        self.model = YOLO(model_path)
        self.logger.info('YOLO model loaded.')

    def process_frame(self, image_msg, odom_msg):
        """
        Returns:
            rx, ry           — robot position (2D)
            frame_rays       — list of (origin_2d, ray_2d) for this frame
            frame_candidates — list of (x, y) new candidates from this frame
        """
        img = self._decode_image(image_msg)
        if img is None:
            return None, None, None, None

        rx, ry, yaw = self._extract_odom(odom_msg)
        self.robot_x.append(rx)
        self.robot_y.append(ry)

        self.processed_frame_count += 1

        # --- Save env frame at regular interval ---
        if self.processed_frame_count % self.env_frame_interval == 0:
            env_path = os.path.join(
                self.env_frames_dir,
                f'env_f{self.processed_frame_count:05d}.jpg'
            )
            cv2.imwrite(env_path, img)

        results = self.model(img, conf=self.confidence, verbose=False)

        frame_rays       = []
        frame_candidates = []

        # --- First pass: save object crops ---
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id          = int(box.cls[0])
                label           = self.model.names[cls_id]

                crop_y1  = max(0, y1)
                crop_y2  = min(img.shape[0], y2)
                crop_x1  = max(0, x1)
                crop_x2  = min(img.shape[1], x2)
                obj_crop = img[crop_y1:crop_y2, crop_x1:crop_x2]
                if obj_crop.size > 0:
                    safe_label = label.replace(' ', '_')
                    obj_path   = os.path.join(
                        self.det_objects_dir,
                        f'f{self.processed_frame_count:05d}_{safe_label}.jpg'
                    )
                    cv2.imwrite(obj_path, obj_crop)

        # --- Second pass: ray casting ---
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id          = int(box.cls[0])
                label           = self.model.names[cls_id]
                px_cx           = (x1 + x2) // 2
                px_cy           = (y1 + y2) // 2

                ray_2d, origin_2d = self._pixel_to_ray_2d(
                    px_cx, px_cy,
                    rx, ry, yaw,
                )

                if ray_2d is None:
                    continue

                frame_rays.append((origin_2d, ray_2d))

                if label not in self.ray_stack:
                    self.ray_stack[label]       = []
                    self.candidate_stack[label] = []

                for prev_origin, prev_ray in self.ray_stack[label]:
                    angle = self._angle_between_rays_2d(ray_2d, prev_ray)

                    if angle < self.min_angle_deg:
                        self.total_skipped += 1
                        continue

                    pt = self._intersect_rays_2d(prev_origin, prev_ray, origin_2d, ray_2d)

                    if pt is None:
                        continue

                    self.candidate_stack[label].append((pt[0], pt[1]))
                    frame_candidates.append((pt[0], pt[1]))
                    self.total_triangulated += 1

                self.ray_stack[label].append((origin_2d, ray_2d))

        return rx, ry, frame_rays, frame_candidates

    def get_object_stack(self):
        self.object_stack = {}
        for label, candidates in self.candidate_stack.items():
            entries = self._cluster_candidates(candidates, label)
            self.object_stack.update(entries)
        return self.object_stack

    def get_all_candidates(self):
        all_candidates = []
        for candidates in self.candidate_stack.values():
            all_candidates.extend(candidates)
        return all_candidates

    def save_outputs(self):
        json_path = os.path.join(self.output_dir, f'{RUN_NAME}_object_stack.json')
        with open(json_path, 'w') as f:
            json.dump(self.object_stack, f, indent=2)
        self.logger.info(f'[{RUN_NAME}] Object stack saved: {json_path}')

        robot_path_data = {'x': self.robot_x, 'y': self.robot_y}
        robot_path_json = os.path.join(self.output_dir, f'{RUN_NAME}_robot_path.json')
        with open(robot_path_json, 'w') as f:
            json.dump(robot_path_data, f)
        self.logger.info(f'[{RUN_NAME}] Robot path saved: {robot_path_json}')

        plot_path = self._save_map_plot()
        self.logger.info(f'[{RUN_NAME}] Map plot saved: {plot_path}')

        self.logger.info(f'[{RUN_NAME}] Total triangulated: {self.total_triangulated}')
        self.logger.info(f'[{RUN_NAME}] Total skipped: {self.total_skipped}')

    # --- Private helpers ---

    def _pixel_to_ray_2d(self, px_cx, px_cy, robot_x, robot_y, yaw):
        """
        Uses tf2_ros Buffer to look up the transform from
        camera_fisheye_front_optical_frame -> odom,
        projects the pixel ray into the odom XY plane (z=0, 2D).

        Returns:
            ray_2d    — normalized 2D direction in odom frame (np.array [x, y])
            origin_2d — 2D camera origin in odom frame (np.array [x, y])
        """
        x_cam       = (px_cx - self.cx) / self.fx
        y_cam       = (px_cy - self.cy) / self.fy
        z_cam       = 1.0
        ray_optical = np.array([x_cam, y_cam, z_cam])
        ray_optical = ray_optical / np.linalg.norm(ray_optical)

        if z_cam <= 0:
            return None, None

        try:
            tf_stamped = self.tf_buffer.lookup_transform(
                'odom',
                'camera_fisheye_front_optical_frame',
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1)
            )
        except Exception as e:
            self.logger.warn(f'TF lookup failed: {e}')
            return None, None

        R = self._quat_to_rotation_matrix(tf_stamped.transform.rotation)
        t = np.array([
            tf_stamped.transform.translation.x,
            tf_stamped.transform.translation.y,
            tf_stamped.transform.translation.z
        ])

        ray_odom_3d = R @ ray_optical

        ray_2d = ray_odom_3d[:2]
        norm   = np.linalg.norm(ray_2d)
        if norm < 1e-6:
            return None, None
        ray_2d = ray_2d / norm

        origin_2d = t[:2]

        return ray_2d, origin_2d

    def _angle_between_rays_2d(self, d1, d2):
        cos_angle = np.clip(np.dot(d1, d2), -1.0, 1.0)
        return np.degrees(np.arccos(cos_angle))

    def _intersect_rays_2d(self, o1, d1, o2, d2):
        """
        2D ray intersection: find closest point between two 2D rays.
        Returns midpoint if both t1 >= 0 and t2 >= 0 (forward direction only).
        """
        A     = np.array([[d1[0], -d2[0]],
                          [d1[1], -d2[1]]])
        b     = o2 - o1
        denom = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
        if abs(denom) < 1e-6:
            return None

        t1 = (b[0] * A[1, 1] - b[1] * A[0, 1]) / denom
        t2 = (A[0, 0] * b[1] - A[1, 0] * b[0]) / denom

        if t1 < 0 or t2 < 0:
            return None

        p1       = o1 + t1 * d1
        p2       = o2 + t2 * d2
        midpoint = (p1 + p2) / 2.0
        return midpoint

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
            self.logger.warn(f'Unsupported encoding: {msg.encoding}')
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

    def _cluster_candidates(self, candidates, label):
        object_entries = {}

        if len(candidates) == 0:
            return object_entries

        pts = np.array(candidates)

        if len(pts) < self.dbscan_min_samples:
            final_x = float(np.median(pts[:, 0]))
            final_y = float(np.median(pts[:, 1]))
            object_entries[label] = {
                'x': round(final_x, 4),
                'y': round(final_y, 4),
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
            cluster_pts = pts[labels_db == cluster_id]
            final_x     = float(np.median(cluster_pts[:, 0]))
            final_y     = float(np.median(cluster_pts[:, 1]))

            if len(unique_clusters) == 1:
                instance_label = label
            else:
                instance_label = f'{label}_{cluster_id + 1}'

            object_entries[instance_label] = {
                'x': round(final_x, 4),
                'y': round(final_y, 4),
                'num_candidates': len(cluster_pts)
            }

        return object_entries

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

            best_dist        = float('inf')
            best_gx, best_gy = None, None
            for gt_label, (gx, gy) in self.ground_truth.items():
                dist = np.sqrt((ox - gx)**2 + (oy - gy)**2)
                if dist < best_dist:
                    best_dist        = dist
                    best_gx, best_gy = gx, gy

            if best_gx is not None:
                plt.plot([ox, best_gx], [oy, best_gy], '--', color=color, linewidth=1.0)
                plt.text((ox + best_gx) / 2, (oy + best_gy) / 2,
                         f'{best_dist:.2f}m', fontsize=8, color=color)

        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.title(f'Semantic Map — {RUN_NAME} — Detected vs Ground Truth')
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

        plot_path = os.path.join(self.output_dir, f'{RUN_NAME}_map.png')
        plt.savefig(plot_path, dpi=150)
        plt.close()
        return plot_path
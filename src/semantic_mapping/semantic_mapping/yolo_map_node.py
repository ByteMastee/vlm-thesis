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

import rclpy


class YoloMapNode:
    def __init__(
        self,
        model_path,
        confidence,
        fx, fy, cx, cy,
        fx_left, fy_left, cx_left, cy_left,
        min_angle_deg,
        dbscan_eps,
        dbscan_min_samples,
        output_dir,
        ground_truth,
        logger,
        tf_buffer,
        run_name,
        min_candidates=3,
        env_frame_interval=20
    ):
        self.confidence         = confidence
        self.fx                 = fx
        self.fy                 = fy
        self.cx                 = cx
        self.cy                 = cy
        self.fx_left            = fx_left
        self.fy_left            = fy_left
        self.cx_left            = cx_left
        self.cy_left            = cy_left
        self.min_angle_deg      = min_angle_deg
        self.dbscan_eps         = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.output_dir         = output_dir
        self.ground_truth       = ground_truth
        self.logger             = logger
        self.tf_buffer          = tf_buffer
        self.run_name           = run_name
        self.min_candidates     = min_candidates
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

    def process_frame(self, image_msg, image_msg_left, odom_msg):
        """
        Process front and left camera frames together.
        Returns:
            rx, ry           — robot position (2D)
            frame_rays       — list of (origin_2d, ray_2d) for this frame
            frame_candidates — list of (x, y) new candidates from this frame
        """
        img_front = self._decode_image(image_msg)
        if img_front is None:
            return None, None, None, None

        img_left = self._decode_image(image_msg_left)
        if img_left is None:
            return None, None, None, None

        rx, ry, yaw = self._extract_odom(odom_msg)
        self.robot_x.append(rx)
        self.robot_y.append(ry)

        self.processed_frame_count += 1

        # --- Save env frame at regular interval (front camera) ---
        if self.processed_frame_count % self.env_frame_interval == 0:
            env_path = os.path.join(
                self.env_frames_dir,
                f'env_f{self.processed_frame_count:05d}.jpg'
            )
            cv2.imwrite(env_path, img_front)

        frame_rays       = []
        frame_candidates = []

        # --- Process front camera ---
        self._process_single_camera(
            img_front, rx, ry, yaw,
            self.fx, self.fy, self.cx, self.cy,
            cam_pitch=0.628, cam_yaw=-0.0255,
            cam_x_offset=0.07, cam_y_offset=0.0,
            frame_rays=frame_rays,
            frame_candidates=frame_candidates
        )

        # --- Process left camera ---
        self._process_single_camera(
            img_left, rx, ry, yaw,
            self.fx_left, self.fy_left, self.cx_left, self.cy_left,
            cam_pitch=0.628, cam_yaw=1.5708,
            cam_x_offset=-0.2, cam_y_offset=0.18,
            frame_rays=frame_rays,
            frame_candidates=frame_candidates
        )

        return rx, ry, frame_rays, frame_candidates

    def _process_single_camera(
        self, img, rx, ry, yaw,
        fx, fy, cx, cy,
        cam_pitch, cam_yaw,
        cam_x_offset, cam_y_offset,
        frame_rays, frame_candidates
    ):
        results = self.model(img, conf=self.confidence, verbose=False)

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
                    px_cx, px_cy, rx, ry, yaw,
                    fx, fy, cx, cy,
                    cam_pitch, cam_yaw,
                    cam_x_offset, cam_y_offset
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

    def get_object_stack(self):
        self.object_stack = {}
        for label, candidates in self.candidate_stack.items():
            if len(candidates) < self.min_candidates:
                self.logger.info(
                    f'[{self.run_name}] Skipping "{label}" — '
                    f'{len(candidates)} candidates < min_candidates={self.min_candidates}'
                )
                continue
            entries = self._cluster_candidates(candidates, label)
            self.object_stack.update(entries)
        return self.object_stack

    def get_all_candidates(self):
        all_candidates = []
        for candidates in self.candidate_stack.values():
            all_candidates.extend(candidates)
        return all_candidates

    def save_outputs(self):
        json_path = os.path.join(self.output_dir, f'{self.run_name}_object_stack.json')
        with open(json_path, 'w') as f:
            json.dump(self.object_stack, f, indent=2)
        self.logger.info(f'[{self.run_name}] Object stack saved: {json_path}')

        robot_path_data = {'x': self.robot_x, 'y': self.robot_y}
        robot_path_json = os.path.join(self.output_dir, f'{self.run_name}_robot_path.json')
        with open(robot_path_json, 'w') as f:
            json.dump(robot_path_data, f)
        self.logger.info(f'[{self.run_name}] Robot path saved: {robot_path_json}')

        plot_path = self._save_map_plot()
        self.logger.info(f'[{self.run_name}] Map plot saved: {plot_path}')

        self.logger.info(f'[{self.run_name}] Total triangulated: {self.total_triangulated}')
        self.logger.info(f'[{self.run_name}] Total skipped: {self.total_skipped}')

    # --- Private helpers ---

    def _pixel_to_ray_2d(
        self, px_cx, px_cy, robot_x, robot_y, yaw,
        fx, fy, cx, cy,
        cam_pitch, cam_yaw,
        cam_x_offset, cam_y_offset
    ):
        # --- Unproject pixel to optical ray ---
        x_cam = (px_cx - cx) / fx
        y_cam = (px_cy - cy) / fy
        z_cam = 1.0
        ray_optical = np.array([x_cam, y_cam, z_cam])
        ray_optical = ray_optical / np.linalg.norm(ray_optical)

        # --- Optical frame to camera body frame ---
        ray_cam_x =  ray_optical[2]
        ray_cam_y = -ray_optical[0]
        ray_cam_z = -ray_optical[1]

        # --- Apply camera pitch (rotation around y-axis) ---
        cos_p = np.cos(cam_pitch)
        sin_p = np.sin(cam_pitch)
        ray_body_x = cos_p * ray_cam_x + sin_p * ray_cam_z
        ray_body_y = ray_cam_y

        # --- Apply camera yaw (rotation around z-axis) ---
        cos_cy = np.cos(cam_yaw)
        sin_cy = np.sin(cam_yaw)
        ray_body_x2 = cos_cy * ray_body_x - sin_cy * ray_body_y
        ray_body_y2 = sin_cy * ray_body_x + cos_cy * ray_body_y

        # --- Project onto horizontal plane and normalize ---
        horiz_norm = np.sqrt(ray_body_x2**2 + ray_body_y2**2)
        if horiz_norm < 1e-6:
            return None, None
        ray_body_x2 = ray_body_x2 / horiz_norm
        ray_body_y2 = ray_body_y2 / horiz_norm

        # --- Apply robot yaw to get odom frame ray ---
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        ray_odom_x = cos_yaw * ray_body_x2 - sin_yaw * ray_body_y2
        ray_odom_y = sin_yaw * ray_body_x2 + cos_yaw * ray_body_y2

        ray_2d = np.array([ray_odom_x, ray_odom_y])
        norm   = np.linalg.norm(ray_2d)
        if norm < 1e-6:
            return None, None
        ray_2d = ray_2d / norm

        # --- Camera origin in odom frame ---
        origin_x  = robot_x + cos_yaw * cam_x_offset - sin_yaw * cam_y_offset
        origin_y  = robot_y + sin_yaw * cam_x_offset + cos_yaw * cam_y_offset
        origin_2d = np.array([origin_x, origin_y])

        return ray_2d, origin_2d

    def _angle_between_rays_2d(self, d1, d2):
        cos_angle = np.clip(np.dot(d1, d2), -1.0, 1.0)
        return np.degrees(np.arccos(cos_angle))

    def _intersect_rays_2d(self, o1, d1, o2, d2):
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

        MAX_DIST = 8.0
        if t1 > MAX_DIST or t2 > MAX_DIST:
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

        if self.ground_truth:
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

        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')

        title = f'Semantic Map — {self.run_name}'
        if self.ground_truth:
            title += ' — Detected vs Ground Truth'
        plt.title(title)

        legend_handles = [
            plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='red',
                    markersize=10, label='Detected'),
            plt.Line2D([0], [0], color='blue', linewidth=1.0, label='Robot path'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green',
                    markersize=8, label='Start'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='red',
                    markersize=8, label='End')
        ]
        if self.ground_truth:
            legend_handles.insert(0, plt.Line2D([0], [0], marker='^', color='w',
                                markerfacecolor='green', markersize=10, label='Ground Truth'))
        plt.legend(handles=legend_handles)

        plt.grid(True)
        plt.axis('equal')

        plot_path = os.path.join(self.output_dir, f'{self.run_name}_map.png')
        plt.savefig(plot_path, dpi=150)
        plt.close()
        return plot_path
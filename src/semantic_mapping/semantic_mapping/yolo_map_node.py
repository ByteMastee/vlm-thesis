import os
import json
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from ultralytics import YOLO

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


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
        logger
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

        self.R_optical_to_base = None
        self.T_cam_offset      = None

        self.ray_stack         = {}
        self.candidate_stack   = {}
        self.object_stack      = {}

        self.robot_x = []
        self.robot_y = []

        self.total_triangulated = 0
        self.total_skipped      = 0

        self.logger.info(f'Loading YOLO model: {model_path}')
        self.model = YOLO(model_path)
        self.logger.info('YOLO model loaded.')

    def set_tf_static(self, tf_msg):
        if self.R_optical_to_base is not None:
            return

        tf_map = {}
        for tf in tf_msg.transforms:
            parent = tf.header.frame_id.lstrip('/')
            child  = tf.child_frame_id.lstrip('/')
            tf_map[(parent, child)] = tf.transform

        key1 = ('base_link', 'camera_fisheye_front_link')
        key2 = ('camera_fisheye_front_link', 'camera_fisheye_front_optical_frame')

        if key1 not in tf_map or key2 not in tf_map:
            return

        T1 = self._tf_to_matrix(tf_map[key1])
        T2 = self._tf_to_matrix(tf_map[key2])
        T_base_to_optical = T1 @ T2
        T_optical_to_base = np.linalg.inv(T_base_to_optical)

        self.R_optical_to_base = T_optical_to_base[:3, :3]
        self.T_cam_offset      = T_base_to_optical[:3, 3]

        self.logger.info('TF static set — R_optical_to_base and T_cam_offset computed.')

    def process_frame(self, image_msg, odom_msg):
        """
        Returns:
            rx, ry          — robot position
            frame_rays      — list of (origin, ray) for this frame
            frame_candidates — list of (x, y) new candidates from this frame
        """
        if self.R_optical_to_base is None:
            self.logger.warn('TF static not set yet — skipping frame.')
            return None, None, None, None

        img = self._decode_image(image_msg)
        if img is None:
            return None, None, None, None

        rx, ry, yaw = self._extract_odom(odom_msg)
        self.robot_x.append(rx)
        self.robot_y.append(ry)

        results = self.model(img, conf=self.confidence, verbose=False)

        frame_rays       = []
        frame_candidates = []

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                label  = self.model.names[cls_id]
                px_cx  = (x1 + x2) // 2
                px_cy  = (y1 + y2) // 2

                origin, ray = self._pixel_to_ray_odom(px_cx, px_cy, rx, ry, yaw)
                # self.logger.info(
                #     f'[{label}] px:({px_cx},{px_cy}) | '
                #     f'origin:({origin[0]:.3f},{origin[1]:.3f},{origin[2]:.3f}) | '
                #     f'ray:({ray[0]:.3f},{ray[1]:.3f},{ray[2]:.3f})'
                # )

                if ray[2] < -0.5:
                    self.logger.info(f'[{label}] Ray filtered — pointing downward: z={ray[2]:.3f}')
                    continue

                frame_rays.append((origin, ray))

                if label not in self.ray_stack:
                    self.ray_stack[label]       = []
                    self.candidate_stack[label] = []

                for prev_origin, prev_ray in self.ray_stack[label]:
                    angle = self._angle_between_rays(ray, prev_ray)

                    if angle < self.min_angle_deg:
                        self.total_skipped += 1
                        continue

                    midpoint = self._closest_approach_midpoint(
                        prev_origin, prev_ray, origin, ray
                    )

                    if midpoint is None:
                        continue

                    self.candidate_stack[label].append((midpoint[0], midpoint[1]))
                    frame_candidates.append((midpoint[0], midpoint[1]))
                    self.total_triangulated += 1

                self.ray_stack[label].append((origin, ray))

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
        json_path = os.path.join(self.output_dir, 'object_stacknew.json')
        with open(json_path, 'w') as f:
            json.dump(self.object_stack, f, indent=2)
        self.logger.info(f'Object stack saved: {json_path}')

        robot_path_data = {'x': self.robot_x, 'y': self.robot_y}
        robot_path_json = os.path.join(self.output_dir, 'robot_pathnew.json')
        with open(robot_path_json, 'w') as f:
            json.dump(robot_path_data, f)
        self.logger.info(f'Robot path saved: {robot_path_json}')

        plot_path = self._save_map_plot()
        self.logger.info(f'Map plot saved: {plot_path}')

        self.logger.info(f'Total triangulated: {self.total_triangulated}')
        self.logger.info(f'Total skipped: {self.total_skipped}')

    # --- Private helpers ---

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

    def _pixel_to_ray_odom(self, px_cx, px_cy, robot_x, robot_y, yaw):
        x = (px_cx - self.cx) / self.fx
        y = (px_cy - self.cy) / self.fy
        z = 1.0
        ray_cam = np.array([x, y, z])
        ray_cam = ray_cam / np.linalg.norm(ray_cam)

        ray_base = self.R_optical_to_base @ ray_cam
        ray_base = ray_base / np.linalg.norm(ray_base)

        c, s = np.cos(yaw), np.sin(yaw)
        R_yaw = np.array([
            [c, -s, 0],
            [s,  c, 0],
            [0,  0, 1]
        ])
        ray_odom = R_yaw @ ray_base
        ray_odom = ray_odom / np.linalg.norm(ray_odom)

        R2d = np.array([[c, -s], [s, c]])
        cam_offset_rotated = R2d @ self.T_cam_offset[:2]
        origin = np.array([
            robot_x + cam_offset_rotated[0],
            robot_y + cam_offset_rotated[1],
            self.T_cam_offset[2]
        ])

        return origin, ray_odom

    def _angle_between_rays(self, d1, d2):
        cos_angle = np.clip(np.dot(d1, d2), -1.0, 1.0)
        return np.degrees(np.arccos(cos_angle))

    def _closest_approach_midpoint(self, o1, d1, o2, d2):
        w0 = o1 - o2
        a  = np.dot(d1, d1)
        b  = np.dot(d1, d2)
        c  = np.dot(d2, d2)
        d  = np.dot(d1, w0)
        e  = np.dot(d2, w0)

        denom = a * c - b * b
        if abs(denom) < 1e-6:
            return None

        t1 = (b * e - c * d) / denom
        t2 = (a * e - b * d) / denom

        if t1 < 0 or t2 < 0:
            return None

        p1       = o1 + t1 * d1
        p2       = o2 + t2 * d2
        midpoint = (p1 + p2) / 2.0
        return midpoint

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

    def _tf_to_matrix(self, transform):
        R = self._quat_to_rotation_matrix(transform.rotation)
        t = np.array([
            transform.translation.x,
            transform.translation.y,
            transform.translation.z
        ])
        T         = np.eye(4)
        T[:3, :3] = R
        T[:3, 3]  = t
        return T

    def _quat_to_rotation_matrix(self, q):
        qx, qy, qz, qw = q.x, q.y, q.z, q.w
        return np.array([
            [1 - 2*(qy**2 + qz**2),  2*(qx*qy - qz*qw),  2*(qx*qz + qy*qw)],
            [2*(qx*qy + qz*qw),  1 - 2*(qx**2 + qz**2),  2*(qy*qz - qx*qw)],
            [2*(qx*qz - qy*qw),  2*(qy*qz + qx*qw),  1 - 2*(qx**2 + qy**2)]
        ])

    def _save_map_plot(self):
        plt.figure(figsize=(10, 10))

        if self.robot_x and self.robot_y:
            plt.plot(self.robot_x, self.robot_y, 'b-', linewidth=1.0, alpha=0.5)
            plt.plot(self.robot_x[0], self.robot_y[0], 'go', markersize=8)
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
        plt.title('Semantic Map — Detected vs Ground Truth')
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

        plot_path = os.path.join(self.output_dir, 'map_plotnew.png')
        plt.savefig(plot_path, dpi=150)
        plt.close()
        return plot_path
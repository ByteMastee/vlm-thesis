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

import torch
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

import tf2_ros
import rclpy


class SAM2MapNode:
    def __init__(
        self,
        checkpoint_path,
        model_cfg,
        fx, fy, cx, cy,
        min_angle_deg,
        dbscan_eps,
        dbscan_min_samples,
        output_dir,
        ground_truth,
        logger,
        tf_buffer,
        run_name,
        env_frame_interval=20,
        points_per_side=8,
        pred_iou_thresh=0.90,
        stability_score_thresh=0.92,
        min_mask_region_area=3000,
        max_mask_area_fraction=0.10,
        max_regions=6
    ):
        self.fx                     = fx
        self.fy                     = fy
        self.cx                     = cx
        self.cy                     = cy
        self.min_angle_deg          = min_angle_deg
        self.dbscan_eps             = dbscan_eps
        self.dbscan_min_samples     = dbscan_min_samples
        self.output_dir             = output_dir
        self.ground_truth           = ground_truth
        self.logger                 = logger
        self.tf_buffer              = tf_buffer
        self.run_name               = run_name
        self.env_frame_interval     = env_frame_interval
        self.max_mask_area_fraction = max_mask_area_fraction
        self.max_regions            = max_regions
        self.min_mask_region_area   = min_mask_region_area

        self.ray_stack         = {}
        self.candidate_stack   = {}
        self.object_stack      = {}
        self.crop_ray_data     = {}

        self.robot_x = []
        self.robot_y = []

        self.total_triangulated    = 0
        self.total_skipped         = 0
        self.processed_frame_count = 0

        # Cross-frame region tracking
        # tracked_regions: list of dicts:
        #   { 'label': str, 'bbox': (x1,y1,x2,y2), 'last_seen': int }
        self.tracked_regions   = []
        self.next_track_id     = 0
        self.track_iou_thresh  = 0.25   # min IoU to match region to existing track
        self.track_max_unseen  = 5      # frames before a track is considered lost

        # Output folders
        self.det_objects_dir = os.path.join(output_dir, 'detections', 'sam2_objects')
        self.env_frames_dir  = os.path.join(output_dir, 'env_frames')
        os.makedirs(self.det_objects_dir, exist_ok=True)
        os.makedirs(self.env_frames_dir,  exist_ok=True)

        # Load SAM2
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.logger.info(f'[{run_name}] Loading SAM2 model on {device}...')
        sam2_model = build_sam2(model_cfg, checkpoint_path, device=device)
        self.mask_generator = SAM2AutomaticMaskGenerator(
            model=sam2_model,
            points_per_side=points_per_side,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            min_mask_region_area=min_mask_region_area,
            crop_n_layers=0
        )
        self.logger.info(f'[{run_name}] SAM2 model loaded.')

    # ------------------------------------------------------------------ #
    #  Cross-frame region tracker                                          #
    # ------------------------------------------------------------------ #

    def _bbox_iou(self, a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter  = (ix2 - ix1) * (iy2 - iy1)
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        return inter / (area_a + area_b - inter + 1e-6)

    def _match_or_create_track(self, bbox):
        """
        Match bbox to an existing tracked region by IoU.
        Returns the track label string.
        Creates a new track if no match found.
        """
        best_iou   = 0.0
        best_idx   = -1

        for i, track in enumerate(self.tracked_regions):
            iou = self._bbox_iou(bbox, track['bbox'])
            if iou > best_iou:
                best_iou = iou
                best_idx = i

        if best_iou >= self.track_iou_thresh:
            # Update existing track
            self.tracked_regions[best_idx]['bbox']      = bbox
            self.tracked_regions[best_idx]['last_seen'] = self.processed_frame_count
            return self.tracked_regions[best_idx]['label']
        else:
            # New track
            label = f'obj_{self.next_track_id:03d}'
            self.next_track_id += 1
            self.tracked_regions.append({
                'label':     label,
                'bbox':      bbox,
                'last_seen': self.processed_frame_count
            })
            return label

    def _prune_lost_tracks(self):
        """Remove tracks not seen for more than track_max_unseen frames."""
        self.tracked_regions = [
            t for t in self.tracked_regions
            if (self.processed_frame_count - t['last_seen']) <= self.track_max_unseen
        ]

    # ------------------------------------------------------------------ #
    #  Main frame processing                                               #
    # ------------------------------------------------------------------ #

    def process_frame(self, image_msg, odom_msg, tf_frame='camera_fisheye_front_optical_frame'):
        img = self._decode_image(image_msg)
        if img is None:
            return None, None, None, None

        rx, ry, yaw = self._extract_odom(odom_msg)
        self.robot_x.append(rx)
        self.robot_y.append(ry)

        self.processed_frame_count += 1

        # Save env frame at regular interval
        if self.processed_frame_count % self.env_frame_interval == 0:
            env_path = os.path.join(
                self.env_frames_dir,
                f'env_f{self.processed_frame_count:05d}.jpg'
            )
            cv2.imwrite(env_path, img)

        # SAM2 region proposals
        img_rgb        = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img.shape[:2]
        total_px       = orig_h * orig_w

        masks   = self.mask_generator.generate(img_rgb)
        regions = []
        for m in masks:
            x, y, w, h = m['bbox']
            area_px     = int(w * h)

            # Area filters
            if area_px / total_px > self.max_mask_area_fraction:
                continue
            if area_px < self.min_mask_region_area:
                continue

            # Aspect ratio filter — reject flat wall/floor shaped masks
            aspect = max(w, h) / (min(w, h) + 1e-6)
            if aspect > 4.0:
                continue

            x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
            regions.append({
                'bbox':  (x1, y1, x2, y2),
                'score': float(m['predicted_iou']),
                'mask':  m['segmentation']
            })

        regions.sort(key=lambda r: r['score'], reverse=True)
        regions = self._nms(regions, iou_thresh=0.4)
        regions = regions[:self.max_regions]

        self.logger.info(
            f'[{self.run_name}] Frame {self.processed_frame_count}: '
            f'{len(masks)} raw masks -> {len(regions)} after filtering'
        )

        # Prune lost tracks
        self._prune_lost_tracks()

        frame_rays       = []
        frame_candidates = []

        for region in regions:
            x1, y1, x2, y2 = region['bbox']
            px_cx = (x1 + x2) // 2
            px_cy = (y1 + y2) // 2

            # Cross-frame tracking — get consistent label for this bbox
            track_label = self._match_or_create_track(region['bbox'])

            # Save crop
            crop_y1       = max(0, y1)
            crop_y2       = min(img.shape[0], y2)
            crop_x1       = max(0, x1)
            crop_x2       = min(img.shape[1], x2)
            obj_crop      = img[crop_y1:crop_y2, crop_x1:crop_x2]
            crop_filename = f'f{self.processed_frame_count:05d}_{track_label}.jpg'
            if obj_crop.size > 0:
                cv2.imwrite(os.path.join(self.det_objects_dir, crop_filename), obj_crop)

            # Ray casting
            ray_2d, origin_2d = self._pixel_to_ray_2d(px_cx, px_cy, rx, ry, yaw, tf_frame)
            if ray_2d is None:
                continue

            # Save ray data per crop for VLM-filtered rebuild
            self.crop_ray_data[crop_filename] = {
                'origin':    origin_2d.tolist(),
                'ray':       ray_2d.tolist(),
                'region_id': track_label,
                'frame':     self.processed_frame_count
            }

            frame_rays.append((origin_2d, ray_2d))

            if track_label not in self.ray_stack:
                self.ray_stack[track_label]       = []
                self.candidate_stack[track_label] = []

            for prev_origin, prev_ray in self.ray_stack[track_label]:
                angle = self._angle_between_rays_2d(ray_2d, prev_ray)
                if angle < self.min_angle_deg:
                    self.total_skipped += 1
                    continue
                pt = self._intersect_rays_2d(prev_origin, prev_ray, origin_2d, ray_2d)
                if pt is None:
                    continue
                self.candidate_stack[track_label].append((pt[0], pt[1]))
                frame_candidates.append((pt[0], pt[1]))
                self.total_triangulated += 1

            self.ray_stack[track_label].append((origin_2d, ray_2d))

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
        json_path = os.path.join(self.output_dir, f'{self.run_name}_vit_object_stack.json')
        with open(json_path, 'w') as f:
            json.dump(self.object_stack, f, indent=2)
        self.logger.info(f'[{self.run_name}] VIT object stack saved: {json_path}')

        ray_data_path = os.path.join(self.output_dir, f'{self.run_name}_crop_ray_data.json')
        with open(ray_data_path, 'w') as f:
            json.dump(self.crop_ray_data, f, indent=2)
        self.logger.info(f'[{self.run_name}] Crop ray data saved: {ray_data_path}')

        robot_path_data = {'x': self.robot_x, 'y': self.robot_y}
        robot_path_json = os.path.join(self.output_dir, f'{self.run_name}_vit_robot_path.json')
        with open(robot_path_json, 'w') as f:
            json.dump(robot_path_data, f)
        self.logger.info(f'[{self.run_name}] VIT robot path saved: {robot_path_json}')

        plot_path = self._save_map_plot()
        self.logger.info(f'[{self.run_name}] VIT map plot saved: {plot_path}')

        self.logger.info(f'[{self.run_name}] Total triangulated: {self.total_triangulated}')
        self.logger.info(f'[{self.run_name}] Total skipped: {self.total_skipped}')

    # ------------------------------------------------------------------ #
    #  NMS                                                                 #
    # ------------------------------------------------------------------ #

    def _nms(self, regions, iou_thresh=0.4):
        keep       = []
        suppressed = set()
        for i in range(len(regions)):
            if i in suppressed:
                continue
            keep.append(regions[i])
            for j in range(i + 1, len(regions)):
                if j in suppressed:
                    continue
                if self._bbox_iou(regions[i]['bbox'], regions[j]['bbox']) > iou_thresh:
                    suppressed.add(j)
        return keep

    # ------------------------------------------------------------------ #
    #  Private helpers                                                     #
    # ------------------------------------------------------------------ #

    def _pixel_to_ray_2d(self, px_cx, px_cy, robot_x, robot_y, yaw, tf_frame):
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
                tf_frame,
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
        ray_2d      = ray_odom_3d[:2]
        norm        = np.linalg.norm(ray_2d)
        if norm < 1e-6:
            return None, None
        ray_2d    = ray_2d / norm
        origin_2d = t[:2]

        return ray_2d, origin_2d

    def _angle_between_rays_2d(self, d1, d2):
        cos_angle = np.clip(np.dot(d1, d2), -1.0, 1.0)
        return np.degrees(np.arccos(cos_angle))

    def _intersect_rays_2d(self, o1, d1, o2, d2):
        A     = np.array([[d1[0], -d2[0]], [d1[1], -d2[1]]])
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
            cluster_pts    = pts[labels_db == cluster_id]
            final_x        = float(np.median(cluster_pts[:, 0]))
            final_y        = float(np.median(cluster_pts[:, 1]))
            instance_label = label if len(unique_clusters) == 1 \
                             else f'{label}_{cluster_id + 1}'
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
        plt.title(f'VIT+VLM Semantic Map — {self.run_name} — Detected vs Ground Truth')
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

        plot_path = os.path.join(self.output_dir, f'{self.run_name}_vit_map.png')
        plt.savefig(plot_path, dpi=150)
        plt.close()
        return plot_path
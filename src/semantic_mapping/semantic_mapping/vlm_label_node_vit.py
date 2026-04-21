import os
import json
import time
import random
import re
import gc
import threading
from collections import defaultdict

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='scipy')

import numpy as np
import torch
from sklearn.cluster import DBSCAN
import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

BASE_OUTPUT_DIR = '/root/UVC_ws/vf_robot_model_ros2/Final_Output/Testing'


class VlmLabelNodeVit(Node):
    def __init__(self):
        super().__init__('vlm_label_node_vit')

        self.declare_parameter('run_name',           'run_01')
        self.declare_parameter('output_dir',         '')
        self.declare_parameter('model_path',         '/root/UVC_ws/models/qwen2.5-vl-3b')
        self.declare_parameter('max_new_tokens',     128)
        self.declare_parameter('env_sample_count',   5)
        self.declare_parameter('min_angle_deg',      3.0)
        self.declare_parameter('dbscan_eps',         1.0)
        self.declare_parameter('dbscan_min_samples', 2)

        self.run_name           = self.get_parameter('run_name').value
        self.model_path         = self.get_parameter('model_path').value
        self.max_new_tokens     = self.get_parameter('max_new_tokens').value
        self.env_sample_count   = self.get_parameter('env_sample_count').value
        self.min_angle_deg      = self.get_parameter('min_angle_deg').value
        self.dbscan_eps         = self.get_parameter('dbscan_eps').value
        self.dbscan_min_samples = self.get_parameter('dbscan_min_samples').value

        output_dir_param = self.get_parameter('output_dir').value
        if output_dir_param:
            self.output_dir = output_dir_param
        else:
            self.output_dir = os.path.join(BASE_OUTPUT_DIR, self.run_name)

        self.det_objects_dir = os.path.join(self.output_dir, 'detections', 'sam2_objects')
        self.env_frames_dir  = os.path.join(self.output_dir, 'env_frames')
        self.vlm_output_path = os.path.join(self.output_dir, f'{self.run_name}_vit_vlm_labels.json')

        self.pipeline_running = False
        self.model            = None
        self.processor        = None

        self.get_logger().info(f'vlm_label_node_vit starting | RUN_NAME: {self.run_name}')
        self.get_logger().info(f'  output_dir        : {self.output_dir}')
        self.get_logger().info(f'  model_path        : {self.model_path}')
        self.get_logger().info(f'  min_angle_deg     : {self.min_angle_deg}')
        self.get_logger().info(f'  dbscan_eps        : {self.dbscan_eps}')
        self.get_logger().info(f'  dbscan_min_samples: {self.dbscan_min_samples}')

        self.srv = self.create_service(
            Trigger, 'run_vlm_pipeline_vit', self._vlm_pipeline_service_cb
        )
        self.notify_client = self.create_client(Trigger, 'vlm_pipeline_done_vit')
        self.get_logger().info(f'[{self.run_name}] VLM VIT pipeline service ready.')

    def _load_model(self):
        self.get_logger().info(f'[{self.run_name}] Loading Qwen2.5-VL processor...')
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.get_logger().info(f'[{self.run_name}] Loading Qwen2.5-VL model (4-bit)...')
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16
            ),
            device_map='cuda:0'
        )
        self.model.eval()
        self.get_logger().info(f'[{self.run_name}] Qwen2.5-VL loaded.')

    def _vlm_pipeline_service_cb(self, request, response):
        if self.pipeline_running:
            response.success = False
            response.message = 'Pipeline already running.'
            return response

        self.pipeline_running = True
        thread = threading.Thread(target=self._run_pipeline_and_notify, daemon=True)
        thread.start()

        response.success = True
        response.message = 'VLM VIT pipeline started.'
        return response

    def _run_pipeline_and_notify(self):
        try:
            success = self._run()
        except Exception as e:
            self.get_logger().error(f'[{self.run_name}] VLM VIT pipeline error: {e}')
            success = False
        finally:
            self.pipeline_running = False

        if not self.notify_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn(f'[{self.run_name}] vlm_pipeline_done_vit not available.')
            return

        notify_request = Trigger.Request()
        future = self.notify_client.call_async(notify_request)
        future.add_done_callback(self._notify_done_cb)

    def _notify_done_cb(self, future):
        try:
            result = future.result()
            if result.success:
                self.get_logger().info(f'[{self.run_name}] ros_node_vit notified successfully.')
            else:
                self.get_logger().warn(f'[{self.run_name}] Notify response: {result.message}')
        except Exception as e:
            self.get_logger().error(f'[{self.run_name}] Notify callback error: {e}')

    def _run(self):
        self._load_model()

        try:
            env_context = self._get_env_context()
            self.get_logger().info(f'[{self.run_name}] Env context: {env_context}')

            results   = {}
            obj_files = sorted([
                f for f in os.listdir(self.det_objects_dir) if f.endswith('.jpg')
            ])

            if not obj_files:
                self.get_logger().warn(f'[{self.run_name}] No SAM2 object crops found.')
                return False

            self.get_logger().info(
                f'[{self.run_name}] Processing {len(obj_files)} SAM2 crops...'
            )

            for obj_file in obj_files:
                obj_path    = os.path.join(self.det_objects_dir, obj_file)
                name_no_ext = os.path.splitext(obj_file)[0]
                # filename format: f00001_obj_000.jpg
                # frame_id = f00001, region_id = obj_000
                underscore_idx = name_no_ext.index('_')
                frame_id       = name_no_ext[:underscore_idx]
                region_id      = name_no_ext[underscore_idx + 1:]

                t_start   = time.time()
                vlm_label = self._get_vlm_label(obj_path=obj_path, env_context=env_context)
                elapsed   = time.time() - t_start
                vlm_label = self._clean_label(vlm_label)

                self.get_logger().info(
                    f'[{self.run_name}] [{obj_file}] VLM: "{vlm_label}" ({elapsed:.2f}s)'
                )

                results[obj_file] = {
                    'vlm_label': vlm_label,
                    'frame_id':  frame_id,
                    'region_id': region_id
                }

                gc.collect()
                torch.cuda.empty_cache()

            with open(self.vlm_output_path, 'w') as f:
                json.dump(results, f, indent=2)
            self.get_logger().info(
                f'[{self.run_name}] VIT VLM labels saved: {self.vlm_output_path}'
            )

            self._build_vlm_object_stack(results)
            return True

        finally:
            self.get_logger().info(f'[{self.run_name}] Unloading VLM model to free VRAM...')
            del self.model
            del self.processor
            self.model     = None
            self.processor = None
            gc.collect()
            torch.cuda.empty_cache()
            self.get_logger().info(f'[{self.run_name}] VLM model unloaded.')

    def _get_env_context(self):
        if not os.path.exists(self.env_frames_dir):
            return 'No environment context available.'

        env_files = sorted([
            f for f in os.listdir(self.env_frames_dir) if f.endswith('.jpg')
        ])

        if not env_files:
            return 'No environment context available.'

        sample = random.sample(env_files, min(self.env_sample_count, len(env_files)))

        env_prompt = (
            'You are a robot perception system analyzing an indoor environment. '
            'Describe this image in 1-2 sentences: what type of room is it, '
            'what is visible, and what major furniture or objects are present. '
            'Be concise and factual.'
        )

        descriptions = []
        for fname in sample:
            img_path = os.path.join(self.env_frames_dir, fname)
            content  = [
                {'type': 'image', 'image': img_path},
                {'type': 'text',  'text': env_prompt}
            ]
            desc = self._query_vlm(content, max_new_tokens=96)
            if desc:
                descriptions.append(desc)
            gc.collect()
            torch.cuda.empty_cache()

        return ' '.join(descriptions) if descriptions else 'No environment context available.'

    def _clean_label(self, label):
        if not label:
            return 'none'
        label = label.lower().strip()
        label = re.sub(r'\b(\w+)\1\b', r'\1', label)
        words   = label.split()
        deduped = [words[0]]
        for w in words[1:]:
            if w != deduped[-1]:
                deduped.append(w)
        return ' '.join(deduped)

    def _get_vlm_label(self, obj_path, env_context):
        object_prompt = (
            f'You are a robot perception system in an indoor environment. '
            f'Environment context: {env_context} '
            f'You are given a cropped image of a segmented region detected by a vision model. '
            f'Task: Provide a short label for the main object shown in the crop. '
            f'Rules: '
            f'1. If the crop clearly shows a real indoor object (furniture, appliance, item), '
            f'reply with a specific short label (e.g. blue chair, wooden table, red sofa). '
            f'2. If the crop is mostly a single flat color (gray, white, beige) with no distinct object, '
            f'reply with the single word: none '
            f'3. If the crop shows a wall, floor, ceiling, shadow, or structural surface, '
            f'reply with the single word: none '
            f'4. If the crop is too small, blurry, or unclear to identify, '
            f'reply with the single word: none '
            f'Reply with the label only. No explanation.'
        )

        content = [
            {'type': 'image', 'image': obj_path},
            {'type': 'text',  'text': object_prompt}
        ]

        result = self._query_vlm(content, max_new_tokens=self.max_new_tokens)
        return result if result else 'none'

    def _query_vlm(self, content_list, max_new_tokens=128):
        try:
            messages = [{'role': 'user', 'content': content_list}]
            text     = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                return_tensors='pt'
            ).to('cuda:0')

            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None
                )

            response = self.processor.decode(
                output[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            ).strip()

            del inputs, output
            gc.collect()
            torch.cuda.empty_cache()

            return response if response else None

        except Exception as e:
            self.get_logger().warn(f'[{self.run_name}] VLM inference error: {e}')
            gc.collect()
            torch.cuda.empty_cache()
            return None

    def _build_vlm_object_stack(self, results):
        ray_data_path = os.path.join(
            self.output_dir, f'{self.run_name}_crop_ray_data.json'
        )
        if not os.path.exists(ray_data_path):
            self.get_logger().warn(
                f'[{self.run_name}] crop_ray_data.json not found — cannot rebuild.'
            )
            return

        with open(ray_data_path, 'r') as f:
            crop_ray_data = json.load(f)

        # Build filtered ray stacks — only VLM-confirmed (non-none) crops
        filtered_ray_stack = defaultdict(list)

        for crop_filename, ray_info in crop_ray_data.items():
            if crop_filename not in results:
                continue
            vlm_label = results[crop_filename]['vlm_label']
            if vlm_label == 'none':
                continue
            origin_2d = np.array(ray_info['origin'])
            ray_2d    = np.array(ray_info['ray'])
            filtered_ray_stack[vlm_label].append((origin_2d, ray_2d))

        # Filter out labels with fewer than 3 rays — not enough for reliable triangulation
        filtered_ray_stack = {
            label: rays
            for label, rays in filtered_ray_stack.items()
            if len(rays) >= 3
        }

        self.get_logger().info(
            f'[{self.run_name}] VLM-confirmed labels after ray count filter: '
            f'{list(filtered_ray_stack.keys())}'
        )

        # Triangulate per VLM label
        filtered_candidates = defaultdict(list)
        total_tri  = 0
        total_skip = 0

        for label, rays in filtered_ray_stack.items():
            for i in range(len(rays)):
                for j in range(i + 1, len(rays)):
                    o1, d1 = rays[i]
                    o2, d2 = rays[j]
                    cos_a  = np.clip(np.dot(d1, d2), -1.0, 1.0)
                    angle  = np.degrees(np.arccos(cos_a))
                    if angle < self.min_angle_deg:
                        total_skip += 1
                        continue
                    A     = np.array([[d1[0], -d2[0]], [d1[1], -d2[1]]])
                    b     = o2 - o1
                    denom = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
                    if abs(denom) < 1e-6:
                        continue
                    t1 = (b[0] * A[1, 1] - b[1] * A[0, 1]) / denom
                    t2 = (A[0, 0] * b[1] - A[1, 0] * b[0]) / denom
                    if t1 < 0 or t2 < 0:
                        continue
                    pt = ((o1 + t1 * d1) + (o2 + t2 * d2)) / 2.0
                    filtered_candidates[label].append((float(pt[0]), float(pt[1])))
                    total_tri += 1

        self.get_logger().info(
            f'[{self.run_name}] Filtered triangulation: {total_tri} pts, '
            f'{total_skip} skipped'
        )

        # DBSCAN cluster per label — keep largest cluster only
        vlm_object_stack = {}

        for label, candidates in filtered_candidates.items():
            if len(candidates) == 0:
                continue

            pts = np.array(candidates)

            if len(pts) < self.dbscan_min_samples:
                vlm_object_stack[label] = {
                    'x':              round(float(np.median(pts[:, 0])), 4),
                    'y':              round(float(np.median(pts[:, 1])), 4),
                    'num_candidates': len(pts),
                    'vlm_label':      label,
                    'region_key':     label
                }
                continue

            db        = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples).fit(pts)
            labels_db = db.labels_
            unique_clusters = set(labels_db) - {-1}

            if not unique_clusters:
                continue

            best_cluster = max(unique_clusters, key=lambda c: (labels_db == c).sum())
            cluster_pts  = pts[labels_db == best_cluster]
            final_x      = round(float(np.median(cluster_pts[:, 0])), 4)
            final_y      = round(float(np.median(cluster_pts[:, 1])), 4)

            vlm_object_stack[label] = {
                'x':              final_x,
                'y':              final_y,
                'num_candidates': len(cluster_pts),
                'vlm_label':      label,
                'region_key':     label
            }
            self.get_logger().info(
                f'[{self.run_name}] "{label}" -> ({final_x}, {final_y}) '
                f'from {len(cluster_pts)} candidates'
            )

        vlm_stack_path = os.path.join(
            self.output_dir, f'{self.run_name}_vit_vlm_object_stack.json'
        )
        with open(vlm_stack_path, 'w') as f:
            json.dump(vlm_object_stack, f, indent=2)
        self.get_logger().info(
            f'[{self.run_name}] VIT VLM object stack saved: {vlm_stack_path} '
            f'({len(vlm_object_stack)} objects)'
        )


def main(args=None):
    rclpy.init(args=args)
    node = VlmLabelNodeVit()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
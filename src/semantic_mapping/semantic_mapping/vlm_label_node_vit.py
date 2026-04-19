import os
import json
import time
import random
import re
import gc
import threading
from collections import Counter

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='scipy')

import cv2
import torch
import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

BASE_OUTPUT_DIR = '/root/UVC_ws/vf_robot_model_ros2/Final_Output/Testing'


class VlmLabelNodeVit(Node):
    def __init__(self):
        super().__init__('vlm_label_node_vit')

        self.declare_parameter('run_name',         'run_01')
        self.declare_parameter('output_dir',       '')
        self.declare_parameter('model_path',       '/root/UVC_ws/models/qwen2.5-vl-3b')
        self.declare_parameter('max_new_tokens',   128)
        self.declare_parameter('env_sample_count', 5)

        self.run_name         = self.get_parameter('run_name').value
        self.model_path       = self.get_parameter('model_path').value
        self.max_new_tokens   = self.get_parameter('max_new_tokens').value
        self.env_sample_count = self.get_parameter('env_sample_count').value

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
        self.get_logger().info(f'  output_dir : {self.output_dir}')
        self.get_logger().info(f'  model_path : {self.model_path}')

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
        # Load model only when pipeline is triggered
        self._load_model()

        try:
            env_context = self._get_env_context()
            self.get_logger().info(f'[{self.run_name}] Env context: {env_context}')

            results   = {}
            obj_files = sorted([f for f in os.listdir(self.det_objects_dir) if f.endswith('.jpg')])

            if not obj_files:
                self.get_logger().warn(f'[{self.run_name}] No SAM2 object crops found.')
                return False

            self.get_logger().info(f'[{self.run_name}] Processing {len(obj_files)} SAM2 crops...')

            for obj_file in obj_files:
                obj_path    = os.path.join(self.det_objects_dir, obj_file)
                name_no_ext = os.path.splitext(obj_file)[0]
                parts       = name_no_ext.split('_', 1)
                frame_id    = parts[0]
                region_id   = parts[1] if len(parts) > 1 else 'region00'

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
            self.get_logger().info(f'[{self.run_name}] VIT VLM labels saved: {self.vlm_output_path}')

            self._build_vlm_object_stack(results)
            return True

        finally:
            # Unload model to free VRAM after pipeline completes
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
            f for f in os.listdir(self.env_frames_dir)
            if f.endswith('.jpg')
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
            return label
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
            f'You are given a cropped image of a detected region from the environment. '
            f'Task: Provide a short label for the object in the crop. '
            f'Rules: '
            f'1. If the crop shows a real indoor object, reply with a specific short label '
            f'(e.g. blue chair, wooden table, dark sofa). '
            f'2. If the crop shows a wall, floor, ceiling, or background, reply with the single word: none '
            f'3. If the crop is unclear, reply with the single word: none '
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
                output = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

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
        vit_stack_path = os.path.join(self.output_dir, f'{self.run_name}_vit_object_stack.json')
        if not os.path.exists(vit_stack_path):
            self.get_logger().warn(
                f'[{self.run_name}] {self.run_name}_vit_object_stack.json not found.'
            )
            return

        with open(vit_stack_path, 'r') as f:
            vit_stack = json.load(f)

        # Collect VLM votes per region key
        vlm_votes = {}
        for entry in results.values():
            region_id = entry['region_id']
            vlm_label = entry['vlm_label']
            if vlm_label == 'none':
                continue
            if region_id not in vlm_votes:
                vlm_votes[region_id] = []
            vlm_votes[region_id].append(vlm_label)

        vlm_object_stack = {}
        for cluster_key, data in vit_stack.items():
            # cluster_key format: region00 or region00_1
            base_region = cluster_key.rsplit('_', 1)[0] if '_' in cluster_key else cluster_key

            if base_region in vlm_votes and vlm_votes[base_region]:
                vote_counts    = Counter(vlm_votes[base_region]).most_common()
                majority_label = vote_counts[0][0]
                if len(vote_counts) > 1 and vote_counts[0][1] == vote_counts[1][1]:
                    majority_label = base_region
            else:
                self.get_logger().info(
                    f'[{self.run_name}] Cluster "{cluster_key}" -> no valid VLM votes, skipping.'
                )
                continue

            vlm_object_stack[majority_label] = {
                'x':              data['x'],
                'y':              data['y'],
                'num_candidates': data['num_candidates'],
                'region_key':     cluster_key,
                'vlm_label':      majority_label
            }

            self.get_logger().info(
                f'[{self.run_name}] Cluster "{cluster_key}" -> VLM: "{majority_label}"'
            )

        vlm_stack_path = os.path.join(self.output_dir, f'{self.run_name}_vit_vlm_object_stack.json')
        with open(vlm_stack_path, 'w') as f:
            json.dump(vlm_object_stack, f, indent=2)
        self.get_logger().info(f'[{self.run_name}] VIT VLM object stack saved: {vlm_stack_path}')


def main(args=None):
    rclpy.init(args=args)
    node = VlmLabelNodeVit()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
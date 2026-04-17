import os
import json
import time
import random
import re
import gc
from collections import Counter

import cv2
import torch
import rclpy
from rclpy.node import Node

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

# --- Run name: must match ros_node.py ---
RUN_NAME = 'run_01'

BASE_OUTPUT_DIR = '/root/UVC_ws/vf_robot_model_ros2/Final_Output/Testing'


class VlmLabelNode(Node):
    def __init__(self):
        super().__init__('vlm_label_node')

        # --- Parameters ---
        self.declare_parameter('output_dir',       os.path.join(BASE_OUTPUT_DIR, RUN_NAME))
        self.declare_parameter('model_path',       '/root/UVC_ws/models/qwen2.5-vl-3b')
        self.declare_parameter('max_new_tokens',   128)
        self.declare_parameter('env_sample_count', 5)

        self.output_dir       = self.get_parameter('output_dir').value
        self.model_path       = self.get_parameter('model_path').value
        self.max_new_tokens   = self.get_parameter('max_new_tokens').value
        self.env_sample_count = self.get_parameter('env_sample_count').value

        # --- Folder paths ---
        self.det_objects_dir = os.path.join(self.output_dir, 'detections', 'objects')
        self.env_frames_dir  = os.path.join(self.output_dir, 'env_frames')
        self.vlm_output_path = os.path.join(self.output_dir, f'{RUN_NAME}_vlm_labels.json')

        self.get_logger().info(f'vlm_label_node starting | RUN_NAME: {RUN_NAME}')
        self.get_logger().info(f'  output_dir       : {self.output_dir}')
        self.get_logger().info(f'  model_path       : {self.model_path}')
        self.get_logger().info(f'  max_new_tokens   : {self.max_new_tokens}')
        self.get_logger().info(f'  env_sample_count : {self.env_sample_count}')

        # --- Load model ---
        self._load_model()

        # --- Run pipeline ---
        self._run()

    # -----------------------------------------------------------------------
    # Model loading
    # -----------------------------------------------------------------------

    def _load_model(self):
        self.get_logger().info('Loading Qwen2.5-VL processor...')
        self.processor = AutoProcessor.from_pretrained(self.model_path)

        self.get_logger().info('Loading Qwen2.5-VL model (4-bit quantization)...')
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16
            ),
            device_map='cuda:0'
        )
        self.model.eval()
        self.get_logger().info('Qwen2.5-VL model loaded.')

    # -----------------------------------------------------------------------
    # Main pipeline
    # -----------------------------------------------------------------------

    def _run(self):
        # Step 1 — Environment context from random env frames
        env_context = self._get_env_context()
        self.get_logger().info(f'Environment context:\n{env_context}')

        # Step 2 — For each object crop, get improved VLM label
        results   = {}
        obj_files = sorted([f for f in os.listdir(self.det_objects_dir) if f.endswith('.jpg')])

        if not obj_files:
            self.get_logger().warn('No object crops found. Run ros_node first.')
            return

        self.get_logger().info(f'[{RUN_NAME}] Processing {len(obj_files)} object crops...')

        for obj_file in obj_files:
            obj_path = os.path.join(self.det_objects_dir, obj_file)

            # Parse frame_id and yolo_label from filename: f00042_chair.jpg
            # Also handles legacy format with confidence: f00042_chair_0.56.jpg
            name_no_ext = os.path.splitext(obj_file)[0]
            frame_id    = name_no_ext.split('_', 1)[0]
            remainder   = name_no_ext.split('_', 1)[1] if '_' in name_no_ext else 'unknown'
            parts = remainder.rsplit('_', 1)
            try:
                float(parts[1])
                remainder = parts[0]
            except (ValueError, IndexError):
                pass
            yolo_label = remainder.replace('_', ' ')

            t_start   = time.time()
            vlm_label = self._get_vlm_label(
                obj_path=obj_path,
                yolo_label=yolo_label,
                env_context=env_context
            )
            elapsed = time.time() - t_start

            vlm_label = self._clean_label(vlm_label)

            if vlm_label == 'none':
                self.get_logger().info(
                    f'[{RUN_NAME}] [{obj_file}] YOLO: "{yolo_label}" -> VLM: REJECTED ({elapsed:.2f}s)'
                )
            else:
                self.get_logger().info(
                    f'[{RUN_NAME}] [{obj_file}] YOLO: "{yolo_label}" -> VLM: "{vlm_label}" ({elapsed:.2f}s)'
                )

            results[obj_file] = {
                'yolo_label'     : yolo_label,
                'vlm_label'      : vlm_label,
                'frame_id'       : frame_id,
                'inference_time' : round(elapsed, 3)
            }

            gc.collect()
            torch.cuda.empty_cache()

        # Step 3 — Save per-crop results
        with open(self.vlm_output_path, 'w') as f:
            json.dump(results, f, indent=2)

        self.get_logger().info(f'[{RUN_NAME}] VLM labels saved: {self.vlm_output_path}')
        self.get_logger().info(f'[{RUN_NAME}] Total objects processed: {len(results)}')

        # Step 4 — Majority vote per YOLO cluster -> vlm_object_stack.json
        self._build_vlm_object_stack(results)

    # -----------------------------------------------------------------------
    # Step 1 — Environment context
    # -----------------------------------------------------------------------

    def _get_env_context(self):
        env_files = sorted([
            f for f in os.listdir(self.env_frames_dir) if f.endswith('.jpg')
        ])

        if not env_files:
            self.get_logger().warn('No env frames found. Env context will be empty.')
            return 'No environment context available.'

        sample = random.sample(env_files, min(self.env_sample_count, len(env_files)))
        self.get_logger().info(f'[{RUN_NAME}] Env frames sampled: {sample}')

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
                self.get_logger().info(f'[{RUN_NAME}] Env [{fname}]: {desc}')
            gc.collect()
            torch.cuda.empty_cache()

        if not descriptions:
            return 'No environment context available.'

        return ' '.join(descriptions)

    # -----------------------------------------------------------------------
    # Label cleanup
    # -----------------------------------------------------------------------

    def _clean_label(self, label):
        """Remove duplicate words and run-together repeated substrings."""
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

    # -----------------------------------------------------------------------
    # Step 2 — Per-object VLM label
    # -----------------------------------------------------------------------

    def _get_vlm_label(self, obj_path, yolo_label, env_context):
        object_prompt = (
            f'You are a robot perception system in an indoor environment. '
            f'Environment context: {env_context} '
            f'YOLO detected an object and labelled it as "{yolo_label}". '
            f'You are given a cropped image of that detected object. '
            f'Task: Provide a short improved label for the object in the crop. '
            f'Rules: '
            f'1. If the crop shows a real indoor object, reply with a specific short label (e.g. blue chair, wooden table, dark sofa). '
            f'2. If the crop is a false detection or shows something impossible indoors (e.g. airplane, outdoor sign), reply with the single word: none '
            f'Reply with the label only. No explanation.'
        )

        image_contents = [
            {'type': 'image', 'image': obj_path},
            {'type': 'text',  'text': object_prompt}
        ]

        result = self._query_vlm(image_contents, max_new_tokens=self.max_new_tokens)
        return result if result else yolo_label

    # -----------------------------------------------------------------------
    # VLM query
    # -----------------------------------------------------------------------

    def _query_vlm(self, content_list, max_new_tokens=128):
        try:
            messages = [{'role': 'user', 'content': content_list}]

            text = self.processor.apply_chat_template(
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
                    max_new_tokens=max_new_tokens
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
            self.get_logger().warn(f'VLM inference error: {e}')
            gc.collect()
            torch.cuda.empty_cache()
            return None

    # -----------------------------------------------------------------------
    # Step 4 — Majority vote per YOLO cluster -> vlm_object_stack.json
    # -----------------------------------------------------------------------

    def _build_vlm_object_stack(self, results):
        yolo_stack_path = os.path.join(self.output_dir, f'{RUN_NAME}_object_stack.json')
        if not os.path.exists(yolo_stack_path):
            self.get_logger().warn(
                f'[{RUN_NAME}] {RUN_NAME}_object_stack.json not found. Skipping vlm_object_stack.'
            )
            return

        with open(yolo_stack_path, 'r') as f:
            yolo_stack = json.load(f)
        self.get_logger().info(f'[{RUN_NAME}] Loaded YOLO object stack: {yolo_stack_path}')

        vlm_votes = {}
        for entry in results.values():
            yolo_label = entry['yolo_label']
            vlm_label  = entry['vlm_label']
            if vlm_label == 'none':
                continue
            if yolo_label not in vlm_votes:
                vlm_votes[yolo_label] = []
            vlm_votes[yolo_label].append(vlm_label)

        vlm_object_stack = {}
        for cluster_key, data in yolo_stack.items():
            parts = cluster_key.rsplit('_', 1)
            if len(parts) == 2 and parts[1].isdigit():
                base_label = parts[0]
            else:
                base_label = cluster_key

            if base_label in vlm_votes and vlm_votes[base_label]:
                majority_label = Counter(vlm_votes[base_label]).most_common(1)[0][0]
            else:
                majority_label = base_label

            suffix  = f'_{parts[1]}' if len(parts) == 2 and parts[1].isdigit() else ''
            new_key = f'{majority_label}{suffix}'

            vlm_object_stack[new_key] = {
                'x'             : data['x'],
                'y'             : data['y'],
                'num_candidates': data['num_candidates'],
                'yolo_label'    : cluster_key,
                'vlm_label'     : majority_label
            }

            self.get_logger().info(
                f'[{RUN_NAME}] Cluster "{cluster_key}" -> VLM majority: "{majority_label}" '
                f'(votes: {len(vlm_votes.get(base_label, []))})'
            )

        vlm_stack_path = os.path.join(self.output_dir, f'{RUN_NAME}_vlm_object_stack.json')
        with open(vlm_stack_path, 'w') as f:
            json.dump(vlm_object_stack, f, indent=2)
        self.get_logger().info(f'[{RUN_NAME}] VLM object stack saved: {vlm_stack_path}')


def main(args=None):
    rclpy.init(args=args)
    node = VlmLabelNode()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
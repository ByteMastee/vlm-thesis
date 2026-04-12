import os
import json
import time
import random

import cv2
import torch
import rclpy
from rclpy.node import Node

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info


class VlmLabelNode(Node):
    def __init__(self):
        super().__init__('vlm_label_node')

        # --- Parameters ---
        self.declare_parameter('output_dir',       '/root/UVC_ws/vf_robot_model_ros2/semantic_mapping_output')
        self.declare_parameter('model_path',       '/root/UVC_ws/models/qwen2.5-vl-3b')
        self.declare_parameter('max_new_tokens',   256)
        self.declare_parameter('env_sample_count', 25)

        self.output_dir       = self.get_parameter('output_dir').value
        self.model_path       = self.get_parameter('model_path').value
        self.max_new_tokens   = self.get_parameter('max_new_tokens').value
        self.env_sample_count = self.get_parameter('env_sample_count').value

        # --- Folder paths ---
        self.det_frames_dir = os.path.join(self.output_dir, 'detections', 'frames')
        self.det_objects_dir = os.path.join(self.output_dir, 'detections', 'objects')
        self.env_frames_dir = os.path.join(self.output_dir, 'env_frames')
        self.vlm_output_path = os.path.join(self.output_dir, 'vlm_labels.json')

        self.get_logger().info('vlm_label_node starting...')
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
        results = {}
        obj_files = sorted(os.listdir(self.det_objects_dir))

        if not obj_files:
            self.get_logger().warn('No object crops found. Run ros_node first.')
            return

        self.get_logger().info(f'Processing {len(obj_files)} object crops...')

        for obj_file in obj_files:
            if not obj_file.endswith('.jpg'):
                continue

            obj_path = os.path.join(self.det_objects_dir, obj_file)

            # Parse frame number from filename: f00042_chair.jpg
            name_no_ext = os.path.splitext(obj_file)[0]
            parts = name_no_ext.split('_', 1)
            frame_id = parts[0]                          # e.g. f00042
            yolo_label = parts[1] if len(parts) > 1 else 'unknown'

            # Find matching detection frame
            det_frame_path = os.path.join(self.det_frames_dir, f'{frame_id}.jpg')
            if not os.path.exists(det_frame_path):
                det_frame_path = None

            t_start = time.time()
            vlm_label = self._get_vlm_label(
                obj_path=obj_path,
                det_frame_path=det_frame_path,
                yolo_label=yolo_label,
                env_context=env_context
            )
            elapsed = time.time() - t_start

            self.get_logger().info(
                f'[{obj_file}] YOLO: "{yolo_label}" → VLM: "{vlm_label}" ({elapsed:.2f}s)'
            )

            results[obj_file] = {
                'yolo_label'     : yolo_label,
                'vlm_label'      : vlm_label,
                'frame_id'       : frame_id,
                'inference_time' : round(elapsed, 3)
            }

        # Step 3 — Save results
        with open(self.vlm_output_path, 'w') as f:
            json.dump(results, f, indent=2)

        self.get_logger().info(f'VLM labels saved: {self.vlm_output_path}')
        self.get_logger().info(f'Total objects processed: {len(results)}')

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

        # Random sample
        sample = random.sample(env_files, min(self.env_sample_count, len(env_files)))
        self.get_logger().info(f'Env frames sampled: {sample}')

        env_prompt = (
            'You are a robot perception system analyzing an indoor environment. '
            'Look at these images from different viewpoints of the same room. '
            'Describe the environment in 2-3 sentences: what type of room is it, '
            'what is the general layout, and what major furniture or objects are present. '
            'Be concise and factual.'
        )

        image_contents = []
        for fname in sample:
            image_contents.append({
                'type': 'image',
                'image': os.path.join(self.env_frames_dir, fname)
            })
        image_contents.append({'type': 'text', 'text': env_prompt})

        return self._query_vlm(image_contents, max_new_tokens=256)

    # -----------------------------------------------------------------------
    # Step 2 — Per-object VLM label
    # -----------------------------------------------------------------------

    def _get_vlm_label(self, obj_path, det_frame_path, yolo_label, env_context):
        object_prompt = (
            f'You are a robot perception system. '
            f'Environment context: {env_context} '
            f'YOLO has detected an object and labelled it as "{yolo_label}". '
            f'You are given two images: '
            f'1) The full camera frame where the object was detected. '
            f'2) A cropped image of the detected object. '
            f'Based on the object crop, the detection frame, and the environment context, '
            f'provide an improved and specific label for this object. '
            f'Reply with only the label — a short noun phrase, no explanation.'
        )

        image_contents = []

        if det_frame_path and os.path.exists(det_frame_path):
            image_contents.append({'type': 'image', 'image': det_frame_path})

        image_contents.append({'type': 'image', 'image': obj_path})
        image_contents.append({'type': 'text',  'text': object_prompt})

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

            return response if response else None

        except Exception as e:
            self.get_logger().warn(f'VLM inference error: {e}')
            return None


def main(args=None):
    rclpy.init(args=args)
    node = VlmLabelNode()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
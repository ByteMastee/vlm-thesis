import os
import json
import time
import base64

import cv2
import numpy as np
import torch
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


class QwenVlmTestNode(Node):
    def __init__(self):
        super().__init__('qwen_vlm_test_node')

        # --- Parameters ---
        self.declare_parameter('image_topic', '/fisheye_front/fisheye_front/image_raw')
        self.declare_parameter('model_path',  '/root/UVC_ws/models/qwen2.5-vl-3b')
        self.declare_parameter('frame_skip',  20)
        self.declare_parameter('max_frames',  60)
        self.declare_parameter('prompt',      'What objects are visible in this image? List each object, its color, and a brief description.')
        self.declare_parameter('output_dir',  '/root/UVC_ws/vf_robot_model_ros2/semantic_mapping_output/qwen_test')
        self.declare_parameter('max_new_tokens', 256)

        self.image_topic    = self.get_parameter('image_topic').value
        self.model_path     = self.get_parameter('model_path').value
        self.frame_skip     = self.get_parameter('frame_skip').value
        self.max_frames     = self.get_parameter('max_frames').value
        self.prompt         = self.get_parameter('prompt').value
        self.output_dir     = self.get_parameter('output_dir').value
        self.max_new_tokens = self.get_parameter('max_new_tokens').value

        # --- Output directory ---
        os.makedirs(self.output_dir, exist_ok=True)
        self.results      = []
        self.results_path = os.path.join(self.output_dir, 'qwen_responses.json')

        # --- State ---
        self.frame_count     = 0
        self.processed_count = 0
        self.is_processing   = False
        self.model_ready     = False

        self.get_logger().info('qwen_vlm_test_node starting...')
        self.get_logger().info(f'  image_topic    : {self.image_topic}')
        self.get_logger().info(f'  model_path     : {self.model_path}')
        self.get_logger().info(f'  frame_skip     : {self.frame_skip}')
        self.get_logger().info(f'  max_frames     : {self.max_frames}')
        self.get_logger().info(f'  max_new_tokens : {self.max_new_tokens}')
        self.get_logger().info(f'  prompt         : "{self.prompt}"')
        self.get_logger().info(f'  output_dir     : {self.output_dir}')

        # --- Load model ---
        self._load_model()

        # --- Subscriber ---
        self.create_subscription(Image, self.image_topic, self.image_cb, 10)
        self.get_logger().info('Ready. Start bag playback now.')

    def _load_model(self):
        self.get_logger().info('Loading Qwen2.5-VL processor...')
        self.processor = AutoProcessor.from_pretrained(self.model_path)

        self.get_logger().info('Loading Qwen2.5-VL model (float16, auto device map)...')
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map='auto'
        )
        self.model.eval()
        self.model_ready = True
        self.get_logger().info('Qwen2.5-VL model loaded successfully.')

    def image_cb(self, msg):
        self.frame_count += 1

        if not self.model_ready:
            return

        if self.frame_count % self.frame_skip != 0:
            return

        if self.processed_count >= self.max_frames:
            return

        if self.is_processing:
            return
        self.is_processing = True

        self.processed_count += 1
        self.get_logger().info(
            f'[Frame {self.frame_count}] Sending to Qwen VLM '
            f'(processed: {self.processed_count}/{self.max_frames})'
        )

        img = self._decode_image(msg)
        if img is None:
            self.get_logger().warn('Failed to decode image — skipping.')
            self.is_processing = False
            return

        # Save the frame image
        frame_img_path = os.path.join(
            self.output_dir, f'frame_{self.frame_count:04d}.jpg'
        )
        cv2.imwrite(frame_img_path, img)

        t_start  = time.time()
        response = self._query_qwen(img)
        elapsed  = time.time() - t_start

        if response:
            self.get_logger().info(
                f'[Frame {self.frame_count}] Response ({elapsed:.2f}s):\n{response}'
            )
            entry = {
                'frame_number'  : self.frame_count,
                'timestamp_sec' : msg.header.stamp.sec,
                'model'         : 'qwen2.5-vl-3b',
                'prompt'        : self.prompt,
                'response'      : response,
                'inference_time': round(elapsed, 3),
                'image_saved'   : frame_img_path
            }
            self.results.append(entry)
            self._save_results()
        else:
            self.get_logger().warn(f'[Frame {self.frame_count}] No response from Qwen VLM.')

        self.is_processing = False

    def _query_qwen(self, img):
        try:
            # Save temp image for qwen_vl_utils
            temp_path = os.path.join(self.output_dir, '_temp_query.jpg')
            cv2.imwrite(temp_path, img)

            messages = [
                {
                    'role': 'user',
                    'content': [
                        {'type': 'image', 'image': temp_path},
                        {'type': 'text',  'text': self.prompt}
                    ]
                }
            ]

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                return_tensors='pt'
            ).to('cuda')

            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens
                )

            response = self.processor.decode(
                output[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            ).strip()

            return response if response else None

        except Exception as e:
            self.get_logger().warn(f'Qwen inference error: {e}')
            return None

    def _save_results(self):
        with open(self.results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        self.get_logger().info(f'Results saved: {self.results_path}')

    def _decode_image(self, msg):
        try:
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
        except Exception as e:
            self.get_logger().warn(f'Image decode error: {e}')
            return None


def main(args=None):
    rclpy.init(args=args)
    node = QwenVlmTestNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
import base64
import json
import os
import time

import cv2
import numpy as np
import requests
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image


class VlmTestNode(Node):
    def __init__(self):
        super().__init__('vlm_test_node')

        # --- Parameters ---
        self.declare_parameter('image_topic',  '/fisheye_front/fisheye_front/image_raw')
        self.declare_parameter('ollama_url',   'http://localhost:11434/api/generate')
        self.declare_parameter('model_name',   'moondream')
        self.declare_parameter('frame_skip',   20)
        self.declare_parameter('max_frames',   140)
        self.declare_parameter('prompt',       '')
        self.declare_parameter('output_dir',   '/root/UVC_ws/vf_robot_model_ros2/semantic_mapping_output/vlm_test3')

        self.image_topic  = self.get_parameter('image_topic').value
        self.ollama_url   = self.get_parameter('ollama_url').value
        self.model_name   = self.get_parameter('model_name').value
        self.frame_skip   = self.get_parameter('frame_skip').value
        self.max_frames   = self.get_parameter('max_frames').value
        self.prompt       = self.get_parameter('prompt').value
        self.output_dir   = self.get_parameter('output_dir').value

        # --- Output directory ---
        os.makedirs(self.output_dir, exist_ok=True)
        self.results      = []
        self.results_path = os.path.join(self.output_dir, 'vlm_responses.json')

        # --- State ---
        self.frame_count     = 0
        self.processed_count = 0
        self.is_processing   = False

        # --- Subscriber ---
        self.create_subscription(Image, self.image_topic, self.image_cb, 10)

        self.get_logger().info('vlm_test_node started.')
        self.get_logger().info(f'  image_topic : {self.image_topic}')
        self.get_logger().info(f'  model       : {self.model_name}')
        self.get_logger().info(f'  frame_skip  : {self.frame_skip}')
        self.get_logger().info(f'  max_frames  : {self.max_frames}')
        self.get_logger().info(f'  prompt      : "{self.prompt}" (empty = image only)')
        self.get_logger().info(f'  output_dir  : {self.output_dir}')
        self.get_logger().info('Start bag playback now.')

    def image_cb(self, msg):
        self.frame_count += 1

        if self.frame_count % self.frame_skip != 0:
            return

        if self.processed_count >= self.max_frames:
            return

        if self.is_processing:
            return
        self.is_processing = True

        self.processed_count += 1
        self.get_logger().info(
            f'[Frame {self.frame_count}] Sending to VLM '
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

        b64_image = self._encode_image_b64(img)

        t_start  = time.time()
        response = self._query_vlm(b64_image)
        elapsed  = time.time() - t_start

        if response:
            self.get_logger().info(
                f'[Frame {self.frame_count}] Response ({elapsed:.2f}s):\n{response}'
            )

            entry = {
                'frame_number'  : self.frame_count,
                'timestamp_sec' : msg.header.stamp.sec,
                'model'         : self.model_name,
                'prompt'        : self.prompt if self.prompt else 'Describe what you see in this image.',
                'response'      : response,
                'inference_time': round(elapsed, 3),
                'image_saved'   : frame_img_path
            }
            self.results.append(entry)
            self._save_results()

        else:
            self.get_logger().warn(f'[Frame {self.frame_count}] No response from VLM.')

        self.is_processing = False

    # --- Helpers ---

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

    def _encode_image_b64(self, img):
        _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buffer).decode('utf-8')

    def _query_vlm(self, b64_image):
        prompt_text = self.prompt if self.prompt else 'Describe what you see in this image.'

        payload = {
            'model'  : self.model_name,
            'prompt' : prompt_text,
            'images' : [b64_image],
            'stream' : False
        }

        try:
            resp = requests.post(
                self.ollama_url,
                json=payload,
                timeout=60
            )
            resp.raise_for_status()
            data = resp.json()

            # --- Debug logging ---
            self.get_logger().info(f'Raw Ollama response: {data}')

            response = data.get('response', '').strip()

            if not response:
                self.get_logger().warn(
                    f'Empty response field. done={data.get("done")} '
                    f'done_reason={data.get("done_reason")} '
                    f'full_data={data}'
                )

            return response if response else None

        except requests.exceptions.Timeout:
            self.get_logger().warn('VLM request timed out.')
            return None
        except requests.exceptions.ConnectionError:
            self.get_logger().warn('Cannot connect to Ollama — is ollama serve running on host?')
            return None
        except Exception as e:
            self.get_logger().warn(f'VLM request error: {e}')
            return None


def main(args=None):
    rclpy.init(args=args)
    node = VlmTestNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
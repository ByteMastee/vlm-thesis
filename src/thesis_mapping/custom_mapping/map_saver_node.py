#!/usr/bin/env python3

from pathlib import Path

import numpy as np
from PIL import Image

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid


class MapSaverNode(Node):
    def __init__(self):
        super().__init__('map_saver_node')

        self.map_received = False
        self.latest_map = None

        self.output_dir = Path('/root/UVC_ws/vf_robot_model_ros2/maps')
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.subscription = self.create_subscription(
            OccupancyGrid,
            '/thesis_mapping/grid_map',
            self.map_callback,
            10
        )

        self.get_logger().info('Map saver node started. Waiting for /thesis_mapping/grid_map ...')

    def map_callback(self, msg: OccupancyGrid):
        self.latest_map = msg
        self.map_received = True
        self.get_logger().info('Map received. Saving now...')
        self.save_map(msg)
        rclpy.shutdown()

    def save_map(self, msg: OccupancyGrid):
        width = msg.info.width
        height = msg.info.height

        data = np.array(msg.data, dtype=np.int16).reshape((height, width))

        image = np.zeros((height, width), dtype=np.uint8)

        image[data == 0] = 254      # free
        image[data == 100] = 0      # occupied
        image[data == -1] = 205     # unknown

        image = np.flipud(image)

        pgm_path = self.output_dir / 'map.pgm'
        yaml_path = self.output_dir / 'map.yaml'

        Image.fromarray(image).save(pgm_path)

        origin_x = msg.info.origin.position.x
        origin_y = msg.info.origin.position.y
        origin_z = 0.0

        yaml_content = f"""image: {pgm_path.name}
resolution: {msg.info.resolution}
origin: [{origin_x}, {origin_y}, {origin_z}]
negate: 0
occupied_thresh: 0.65
free_thresh: 0.196
"""

        yaml_path.write_text(yaml_content)

        self.get_logger().info(f'Saved map image: {pgm_path}')
        self.get_logger().info(f'Saved map metadata: {yaml_path}')


def main(args=None):
    rclpy.init(args=args)
    node = MapSaverNode()
    rclpy.spin(node)
    node.destroy_node()


if __name__ == '__main__':
    main()
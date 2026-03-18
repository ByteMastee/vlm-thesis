#!/usr/bin/env python3

import yaml
import math
import numpy as np
from pathlib import Path
from PIL import Image

import rclpy
from rclpy.node import Node

from gazebo_msgs.msg import ModelStates


class SemanticGTNode(Node):

    def __init__(self):
        super().__init__('semantic_gt_node')

        # PATHS
        self.map_dir = Path('/root/UVC_ws/vf_robot_model_ros2/maps')
        self.map_yaml = self.map_dir / 'map.yaml'
        self.map_pgm = self.map_dir / 'map.pgm'

        # CLASS MAP
        self.class_dict = {
            "wall": 1,
            "chair": 2,
            "table": 3,
            "cabinet": 4,
            "bed": 5
        }

        # FOOTPRINT SIZE (meters)
        self.size_dict = {
            "chair": (0.6, 0.6),
            "table": (1.2, 0.8),
            "cabinet": (1.0, 0.5),
            "bed": (2.0, 1.5),
            "wall": (3.0, 0.3)
        }

        # LOAD OCCUPANCY MAP
        self.load_map()

        # semantic grid
        self.semantic_grid = np.zeros_like(self.occ_grid, dtype=np.uint8)

        # SUBSCRIBE GAZEBO TRUTH
        self.sub = self.create_subscription(
            ModelStates,
            '/gazebo/model_states',
            self.model_callback,
            10
        )

        self.received_once = False

        self.get_logger().info("Semantic GT node started")


    def load_map(self):

        with open(self.map_yaml) as f:
            meta = yaml.safe_load(f)

        self.resolution = meta["resolution"]
        self.origin_x = meta["origin"][0]
        self.origin_y = meta["origin"][1]

        img = Image.open(self.map_pgm)
        img = np.array(img)

        # convert to occupancy style
        self.occ_grid = np.zeros_like(img, dtype=np.int8)
        self.occ_grid[img == 0] = 100
        self.occ_grid[img == 254] = 0
        self.occ_grid[img == 205] = -1

        self.height, self.width = self.occ_grid.shape

        self.get_logger().info(f"Map loaded size = {self.width} x {self.height}")

    def world_to_grid(self, x, y):

        gx = int((x - self.origin_x) / self.resolution)
        gy = int((y - self.origin_y) / self.resolution)

        gy = self.height - gy

        return gx, gy


    def draw_rectangle(self, cx, cy, yaw, sx, sy, class_id):

        halfx = sx / 2.0
        halfy = sy / 2.0

        corners = [
            (-halfx, -halfy),
            (halfx, -halfy),
            (halfx, halfy),
            (-halfx, halfy)
        ]

        world_pts = []

        for px, py in corners:

            wx = cx + px * math.cos(yaw) - py * math.sin(yaw)
            wy = cy + px * math.sin(yaw) + py * math.cos(yaw)

            gx, gy = self.world_to_grid(wx, wy)

            world_pts.append((gx, gy))

        xs = [p[0] for p in world_pts]
        ys = [p[1] for p in world_pts]

        xmin = max(min(xs), 0)
        xmax = min(max(xs), self.width - 1)

        ymin = max(min(ys), 0)
        ymax = min(max(ys), self.height - 1)

        for x in range(xmin, xmax):
            for y in range(ymin, ymax):

                if self.occ_grid[y, x] == 100:
                    self.semantic_grid[y, x] = class_id

    def classify(self, name):

        lname = name.lower()

        for key in self.class_dict:
            if key in lname:
                return self.class_dict[key], key

        return None, None


    def model_callback(self, msg):

        if self.received_once:
            return

        self.get_logger().info("Generating semantic GT map...")

        for i, name in enumerate(msg.name):

            class_id, label = self.classify(name)

            if class_id is None:
                continue

            pose = msg.pose[i]

            x = pose.position.x
            y = pose.position.y

            q = pose.orientation

            yaw = math.atan2(
                2 * (q.w * q.z + q.x * q.y),
                1 - 2 * (q.y * q.y + q.z * q.z)
            )

            sx, sy = self.size_dict[label]

            self.draw_rectangle(x, y, yaw, sx, sy, class_id)

        self.save_outputs()

        self.received_once = True

        self.get_logger().info("Semantic GT created successfully")


    def save_outputs(self):

        np.save(self.map_dir / "semantic_map.npy", self.semantic_grid)

        vis = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        color_map = {
            1: (255, 0, 0),
            2: (0, 255, 0),
            3: (0, 0, 255),
            4: (255, 255, 0),
            5: (255, 0, 255)
        }

        for cid, col in color_map.items():
            vis[self.semantic_grid == cid] = col

        Image.fromarray(vis).save(self.map_dir / "semantic_map.png")

        self.get_logger().info("Saved semantic_map.npy")
        self.get_logger().info("Saved semantic_map.png")


def main(args=None):
    rclpy.init(args=args)
    node = SemanticGTNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
#!/usr/bin/env python3
"""
map_saver_periodic.py

Saves the current /map to disk every N seconds automatically.

Place at:   uvc1_gazebo/uvc1_gazebo/map_saver_periodic.py

Add to setup.py console_scripts:
    'map_saver_periodic = uvc1_gazebo.map_saver_periodic:main',

Authors: Pravin Oli  /  generated with Claude
"""

import os
import subprocess
import datetime

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy


class MapSaverPeriodic(Node):

    def __init__(self):
        super().__init__("map_saver_periodic")

        self.declare_parameter("map_save_dir",      os.path.expanduser("~/explored_maps"))
        self.declare_parameter("map_save_interval",  30.0)

        self.map_save_dir      = self.get_parameter("map_save_dir").value
        self.map_save_interval = self.get_parameter("map_save_interval").value

        os.makedirs(self.map_save_dir, exist_ok=True)

        self.get_logger().info(f"Map save dir    : {self.map_save_dir}")
        self.get_logger().info(f"Save interval   : {self.map_save_interval}s")

        self._map_received = False

        map_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.create_subscription(OccupancyGrid, "/map", self._map_cb, map_qos)
        self.create_timer(self.map_save_interval, self._save_map)

        self.get_logger().info("Waiting for first /map message...")

    def _map_cb(self, msg: OccupancyGrid):
        if not self._map_received:
            self.get_logger().info("/map received — periodic saving is active.")
        self._map_received = True

    def _save_map(self):
        if not self._map_received:
            self.get_logger().info("No /map yet — skipping save.")
            return

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base_path = os.path.join(self.map_save_dir, f"map_{timestamp}")

        self.get_logger().info(f"Saving map → {base_path}")
        try:
            result = subprocess.run(
                [
                    "ros2", "run", "nav2_map_server", "map_saver_cli",
                    "-f", base_path,
                    "--ros-args", "-p", "save_map_timeout:=5.0",
                ],
                timeout=15,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                self.get_logger().info(f"Saved: {base_path}.pgm + .yaml")
            else:
                self.get_logger().error(f"Save failed:\n{result.stderr}")
        except subprocess.TimeoutExpired:
            self.get_logger().error("map_saver_cli timed out.")


def main(args=None):
    rclpy.init(args=args)
    node = MapSaverPeriodic()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

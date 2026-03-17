#!/usr/bin/env python3

import math
import struct

import numpy as np

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PointStamped
from tf2_ros import Buffer, TransformListener, TransformException
from tf2_geometry_msgs.tf2_geometry_msgs import do_transform_point


class NewMapNode(Node):
    def __init__(self):
        super().__init__('new_map_node')

        self.latest_camera_info = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.map_resolution = 0.10
        self.map_width = 600
        self.map_height = 600
        self.map_origin_x = -30.0
        self.map_origin_y = -30.0

        self.min_valid_depth = 0.2
        self.max_valid_depth = 8.0

        # Top-camera world-height filter
        self.min_obstacle_height = 0.05
        self.max_obstacle_height = 1.50

        self.u_step = 15
        self.v_step = 15

        self.grid = np.full((self.map_height, self.map_width), -1, dtype=np.int8)

        self.map_pub = self.create_publisher(
            OccupancyGrid,
            '/thesis_mapping/grid_map',
            10
        )

        self.info_sub = self.create_subscription(
            CameraInfo,
            '/d435i/depth/d435i_depth/camera_info',
            self.info_callback,
            10
        )

        self.depth_sub = self.create_subscription(
            Image,
            '/d435i/depth/d435i_depth/depth/image_raw',
            self.depth_callback,
            10
        )

        self.get_logger().info('ne_map_node started with TF-based robot pose and top camera.')

    def info_callback(self, msg: CameraInfo):
        self.latest_camera_info = msg

    def world_to_grid(self, x, y):
        gx = int((x - self.map_origin_x) / self.map_resolution)
        gy = int((y - self.map_origin_y) / self.map_resolution)

        if gx < 0 or gx >= self.map_width or gy < 0 or gy >= self.map_height:
            return None

        return gx, gy

    def bresenham(self, x0, y0, x1, y1):
        cells = []

        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        x, y = x0, y0

        while True:
            cells.append((x, y))
            if x == x1 and y == y1:
                break

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

        return cells

    def publish_map(self, stamp):
        msg = OccupancyGrid()
        msg.header.stamp = stamp
        msg.header.frame_id = 'odom'

        msg.info.resolution = self.map_resolution
        msg.info.width = self.map_width
        msg.info.height = self.map_height
        msg.info.origin.position.x = self.map_origin_x
        msg.info.origin.position.y = self.map_origin_y
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.w = 1.0

        msg.data = self.grid.flatten().tolist()
        self.map_pub.publish(msg)

    def get_robot_cell_from_tf(self):
        try:
            base_tf = self.tf_buffer.lookup_transform(
                'odom',
                'base_link',
                rclpy.time.Time()
            )
        except TransformException as ex:
            self.get_logger().warn(f'Robot TF lookup failed (odom <- base_link): {ex}')
            return None

        robot_x = base_tf.transform.translation.x
        robot_y = base_tf.transform.translation.y

        robot_cell = self.world_to_grid(robot_x, robot_y)
        if robot_cell is None:
            self.get_logger().warn('Robot position is outside map bounds.')
            return None

        return robot_cell

    def depth_callback(self, msg: Image):
        if self.latest_camera_info is None:
            self.get_logger().warn('CameraInfo not received yet.')
            return

        if msg.encoding != '32FC1':
            self.get_logger().warn(f'Unsupported depth encoding: {msg.encoding}')
            return

        robot_cell = self.get_robot_cell_from_tf()
        if robot_cell is None:
            return

        try:
            camera_tf = self.tf_buffer.lookup_transform(
                'odom',
                msg.header.frame_id,
                rclpy.time.Time()
            )
        except TransformException as ex:
            self.get_logger().warn(f'Camera TF lookup failed (odom <- {msg.header.frame_id}): {ex}')
            return

        fx = self.latest_camera_info.k[0]
        fy = self.latest_camera_info.k[4]
        cx = self.latest_camera_info.k[2]
        cy = self.latest_camera_info.k[5]

        width = msg.width
        height = msg.height

        occupied_updates = 0
        free_updates = 0
        valid_hits = 0

        for v in range(0, height, self.v_step):
            for u in range(0, width, self.u_step):
                byte_index = v * msg.step + u * 4

                if byte_index + 4 > len(msg.data):
                    continue

                depth = struct.unpack_from('<f', msg.data, byte_index)[0]

                if (
                    not math.isfinite(depth)
                    or depth < self.min_valid_depth
                    or depth > self.max_valid_depth
                ):
                    continue

                x_cam = (u - cx) * depth / fx
                y_cam = (v - cy) * depth / fy
                z_cam = depth

                point_camera = PointStamped()
                point_camera.header = msg.header
                point_camera.point.x = x_cam
                point_camera.point.y = y_cam
                point_camera.point.z = z_cam

                point_odom = do_transform_point(point_camera, camera_tf)

                z_world = point_odom.point.z
                if z_world < self.min_obstacle_height or z_world > self.max_obstacle_height:
                    continue

                hit_cell = self.world_to_grid(point_odom.point.x, point_odom.point.y)
                if hit_cell is None:
                    continue

                valid_hits += 1

                ray_cells = self.bresenham(
                    robot_cell[0], robot_cell[1],
                    hit_cell[0], hit_cell[1]
                )

                for free_cell in ray_cells[:-1]:
                    fxg, fyg = free_cell
                    if self.grid[fyg, fxg] == -1:
                        self.grid[fyg, fxg] = 0
                        free_updates += 1

                hx, hy = hit_cell
                if self.grid[hy, hx] != 100:
                    self.grid[hy, hx] = 100
                    occupied_updates += 1

        self.publish_map(msg.header.stamp)
        self.get_logger().info(
            f'Frame updates -> valid_hits: {valid_hits}, free: {free_updates}, occupied: {occupied_updates}'
        )


def main(args=None):
    rclpy.init(args=args)
    node = NewMapNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
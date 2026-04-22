#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np


class FisheyeRectifyNode(Node):
    def __init__(self):
        super().__init__('fisheye_rectify_node')

        self.bridge = CvBridge()

        # Calibration parameters from camera_info (equidistant model)
        self.K = np.array([
            [271.7130252551282, 0.0,               320.79340187886527],
            [0.0,               272.1434051421597, 226.64923232430766],
            [0.0,               0.0,               1.0]
        ], dtype=np.float64)

        self.D = np.array([
            [-0.05268338601520044],
            [ 0.004889702830369265],
            [-0.0031745403716575385],
            [ 0.00015737212611983246]
        ], dtype=np.float64)

        self.P = np.array([
            [252.0830841064453, 0.0,               321.147701029975,  0.0],
            [0.0,               260.4607849121094, 223.13973599103883, 0.0],
            [0.0,               0.0,               1.0,               0.0]
        ], dtype=np.float64)

        self.image_size = (640, 480)

        # Precompute undistortion maps
        self.K_new = self.P[:3, :3]
        self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(
            self.K,
            self.D,
            np.eye(3),
            self.K_new,
            self.image_size,
            cv2.CV_16SC2
        )

        self.get_logger().info('Undistortion maps computed.')

        self.sub = self.create_subscription(
            Image,
            '/fisheye_front/fisheye_front/image_raw',
            self.image_callback,
            10
        )

        self.pub = self.create_publisher(
            Image,
            '/fisheye_front/fisheye_front/image_rect',
            10
        )

    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        rectified = cv2.remap(cv_image, self.map1, self.map2, cv2.INTER_LINEAR)
        rect_msg = self.bridge.cv2_to_imgmsg(rectified, encoding='rgb8')
        rect_msg.header = msg.header
        self.pub.publish(rect_msg)


def main(args=None):
    rclpy.init(args=args)
    node = FisheyeRectifyNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
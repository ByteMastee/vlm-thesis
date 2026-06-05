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

        # --- Front camera calibration ---
        self.K_front = np.array([
            [271.7130252551282, 0.0,               320.79340187886527],
            [0.0,               272.1434051421597, 226.64923232430766],
            [0.0,               0.0,               1.0]
        ], dtype=np.float64)

        self.D_front = np.array([
            [-0.05268338601520044],
            [ 0.004889702830369265],
            [-0.0031745403716575385],
            [ 0.00015737212611983246]
        ], dtype=np.float64)

        self.P_front = np.array([
            [252.0830841064453, 0.0,               321.147701029975,  0.0],
            [0.0,               260.4607849121094, 223.13973599103883, 0.0],
            [0.0,               0.0,               1.0,               0.0]
        ], dtype=np.float64)

        # --- Left camera calibration ---
        self.K_left = np.array([
            [274.45916376464004, 0.0,               329.2988823957579],
            [0.0,                275.0141620205343, 227.55314053282927],
            [0.0,                0.0,               1.0]
        ], dtype=np.float64)

        self.D_left = np.array([
            [-0.05836727332135975],
            [ 0.012035495204542137],
            [-0.007326628722429022],
            [ 0.0008980731607595731]
        ], dtype=np.float64)

        self.P_left = np.array([
            [258.6033630371094, 0.0,               330.8103127361028,  0.0],
            [0.0,               262.9443054199219, 221.3089734609366,  0.0],
            [0.0,               0.0,               1.0,                0.0]
        ], dtype=np.float64)

        self.image_size = (640, 480)

        # --- Precompute undistortion maps ---
        self.K_new_front = self.P_front[:3, :3]
        self.map1_front, self.map2_front = cv2.fisheye.initUndistortRectifyMap(
            self.K_front, self.D_front, np.eye(3),
            self.K_new_front, self.image_size, cv2.CV_16SC2
        )

        self.K_new_left = self.P_left[:3, :3]
        self.map1_left, self.map2_left = cv2.fisheye.initUndistortRectifyMap(
            self.K_left, self.D_left, np.eye(3),
            self.K_new_left, self.image_size, cv2.CV_16SC2
        )

        self.get_logger().info('Front and left undistortion maps computed.')

        # --- Subscriptions ---
        self.sub_front = self.create_subscription(
            Image,
            '/fisheye_front/fisheye_front/image_raw',
            self.image_callback_front,
            10
        )

        self.sub_left = self.create_subscription(
            Image,
            '/fisheye_left/fisheye_left/image_raw',
            self.image_callback_left,
            10
        )

        # --- Publishers ---
        self.pub_front = self.create_publisher(
            Image,
            '/fisheye_front/fisheye_front/image_rect',
            10
        )

        self.pub_left = self.create_publisher(
            Image,
            '/fisheye_left/fisheye_left/image_rect',
            10
        )

    def image_callback_front(self, msg):
        cv_image  = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        rectified = cv2.remap(cv_image, self.map1_front, self.map2_front, cv2.INTER_LINEAR)
        rect_msg  = self.bridge.cv2_to_imgmsg(rectified, encoding='rgb8')
        rect_msg.header = msg.header
        self.pub_front.publish(rect_msg)

    def image_callback_left(self, msg):
        cv_image  = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        rectified = cv2.remap(cv_image, self.map1_left, self.map2_left, cv2.INTER_LINEAR)
        rect_msg  = self.bridge.cv2_to_imgmsg(rectified, encoding='rgb8')
        rect_msg.header = msg.header
        self.pub_left.publish(rect_msg)


def main(args=None):
    rclpy.init(args=args)
    node = FisheyeRectifyNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
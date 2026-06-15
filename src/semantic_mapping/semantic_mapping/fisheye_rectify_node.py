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

        self.bridge   = CvBridge()
        self.is_ready = False
        self.map1     = None
        self.map2     = None

        self.sub_info = self.create_subscription(
            CameraInfo,
            '/fisheye_front/camera_info',
            self.cam_info_cb,
            10
        )

        self.sub_image = self.create_subscription(
            Image,
            '/fisheye_front/image_raw',
            self.image_callback,
            10
        )

        self.pub = self.create_publisher(
            Image,
            '/fisheye_front/image_rect',
            10
        )

        self.get_logger().info('Waiting for camera_info...')

    def cam_info_cb(self, msg):
        if self.is_ready:
            return

        image_size = (msg.width, msg.height)

        K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        D = np.array(msg.d, dtype=np.float64).reshape(-1, 1)
        P = np.array(msg.p, dtype=np.float64).reshape(3, 4)
        K_new = P[:3, :3]

        distortion_model = msg.distortion_model
        self.get_logger().info(f'Distortion model: {distortion_model}')

        try:
            if distortion_model == 'equidistant':
                self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(
                    K, D, np.eye(3), K_new, image_size, cv2.CV_16SC2
                )
            elif distortion_model in ('plumb_bob', 'rational_polynomial'):
                D_flat = np.array(msg.d, dtype=np.float64)
                self.map1, self.map2 = cv2.initUndistortRectifyMap(
                    K, D_flat, np.eye(3), K_new, image_size, cv2.CV_16SC2
                )
            else:
                self.get_logger().warn(
                    f'Unknown distortion model: {distortion_model} — '
                    f'attempting equidistant.'
                )
                self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(
                    K, D, np.eye(3), K_new, image_size, cv2.CV_16SC2
                )
        except Exception as e:
            self.get_logger().error(f'Failed to compute undistortion maps: {e}')
            return

        self.is_ready = True
        self.get_logger().info(
            f'Camera calibrated — '
            f'fx:{K[0,0]:.4f} fy:{K[1,1]:.4f} '
            f'cx:{K[0,2]:.4f} cy:{K[1,2]:.4f} '
            f'size:{image_size} model:{distortion_model}'
        )
        self.get_logger().info('Undistortion maps computed.')

    def image_callback(self, msg):
        if not self.is_ready:
            return

        cv_image  = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        rectified = cv2.remap(cv_image, self.map1, self.map2, cv2.INTER_LINEAR)
        rect_msg  = self.bridge.cv2_to_imgmsg(rectified, encoding='rgb8')
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
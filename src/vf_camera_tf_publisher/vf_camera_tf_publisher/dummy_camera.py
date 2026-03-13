#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

# Import your messages and services
from vfmessages.msg import ChannelValues
from vfmessages.srv import CalibrateCamera


def main():
    rclpy.init()
    node = Node("vfmessages_dummy_test")

    # Test publishing a message
    msg = ChannelValues()
    msg.channel_1 = 123
    msg.channel_2 = 456
    node.get_logger().info(
        f"Test message: channel_1={msg.channel_1}, channel_2={msg.channel_2}"
    )

    # Test creating a service request
    request = CalibrateCamera.Request()
    request.start_calibration = True
    node.get_logger().info(
        f"Test service request: start_calibration={request.start_calibration}"
    )

    rclpy.shutdown()


if __name__ == "__main__":
    main()

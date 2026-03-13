#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from vfmessages.msg import UltraSound
from sensor_msgs.msg import Range


class Ultrasound(Node):
    def __init__(self):
        super().__init__("ultrasound")  # Node name
        self.get_logger().info("Ultrasound: initializing...")

        # Publisher
        self.ultrasound_publisher = self.create_publisher(UltraSound, "/esp/range", 1)

        # Subscribers
        self.sensor_ids = {
            "front_left": 0,
            "front_right": 1,
            "right": 2,
            "rear": 3,
            "left": 4,
        }

        topics = [
            "/ultrasound/front_left",
            "/ultrasound/front_right",
            "/ultrasound/right",
            "/ultrasound/rear",
            "/ultrasound/left",
        ]

        self.subs = []
        for topic in topics:
            sub = self.create_subscription(Range, topic, self.ultrasound_callback, 1)
            self.subs.append(sub)

    def ultrasound_callback(self, msg: Range):
        # Map frame_id to sensor code
        sensor_name = msg.header.frame_id
        code = self.sensor_ids.get(sensor_name, -1)

        vfm = UltraSound()
        vfm.code = code
        vfm.range = msg.range

        self.ultrasound_publisher.publish(vfm)


def main(args=None):
    rclpy.init(args=args)
    node = Ultrasound()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

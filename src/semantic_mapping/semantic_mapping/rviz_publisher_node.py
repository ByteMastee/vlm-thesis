from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from builtin_interfaces.msg import Duration


class RvizPublisherNode:
    def __init__(self, logger):
        self.logger = logger

    def build_marker_array(self, object_stack, ground_truth, robot_path, clock):
        marker_array = MarkerArray()
        marker_id    = 0

        # --- Detected objects ---
        for label, data in object_stack.items():
            ox = data['x']
            oy = data['y']

            sphere          = self._make_sphere_marker(
                marker_id, ox, oy, 'detected',
                ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0), clock
            )
            marker_array.markers.append(sphere)
            marker_id += 1

            text            = self._make_text_marker(
                marker_id, ox, oy, f'Det: {label}\n({ox:.2f},{oy:.2f})',
                'detected_labels', ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0), clock
            )
            marker_array.markers.append(text)
            marker_id += 1

        # --- Ground truth objects ---
        for label, (gx, gy) in ground_truth.items():
            sphere          = self._make_sphere_marker(
                marker_id, gx, gy, 'ground_truth',
                ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0), clock
            )
            marker_array.markers.append(sphere)
            marker_id += 1

            text            = self._make_text_marker(
                marker_id, gx, gy, f'GT: {label}\n({gx},{gy})',
                'gt_labels', ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0), clock
            )
            marker_array.markers.append(text)
            marker_id += 1

        # --- Robot trajectory ---
        if robot_path is not None:
            traj                    = Marker()
            traj.header.frame_id    = 'odom'
            traj.header.stamp       = clock.now().to_msg()
            traj.ns                 = 'trajectory'
            traj.id                 = marker_id
            traj.type               = Marker.LINE_STRIP
            traj.action             = Marker.ADD
            traj.scale.x            = 0.05
            traj.color              = ColorRGBA(r=0.0, g=0.5, b=1.0, a=0.8)
            traj.pose.orientation.w = 1.0
            traj.lifetime           = Duration(sec=0)

            for px, py in zip(robot_path['x'], robot_path['y']):
                pt   = Point()
                pt.x = px
                pt.y = py
                pt.z = 0.0
                traj.points.append(pt)

            marker_array.markers.append(traj)
            marker_id += 1

        self.logger.info(f'Marker array built — {len(marker_array.markers)} markers.')
        return marker_array

    # --- Private helpers ---

    def _make_sphere_marker(self, marker_id, x, y, ns, color, clock):
        m                    = Marker()
        m.header.frame_id    = 'odom'
        m.header.stamp       = clock.now().to_msg()
        m.ns                 = ns
        m.id                 = marker_id
        m.type               = Marker.SPHERE
        m.action             = Marker.ADD
        m.pose.position.x    = float(x)
        m.pose.position.y    = float(y)
        m.pose.position.z    = 0.0
        m.pose.orientation.w = 1.0
        m.scale.x            = 0.2
        m.scale.y            = 0.2
        m.scale.z            = 0.2
        m.color              = color
        m.lifetime           = Duration(sec=0)
        return m

    def _make_text_marker(self, marker_id, x, y, text, ns, color, clock):
        m                    = Marker()
        m.header.frame_id    = 'odom'
        m.header.stamp       = clock.now().to_msg()
        m.ns                 = ns
        m.id                 = marker_id
        m.type               = Marker.TEXT_VIEW_FACING
        m.action             = Marker.ADD
        m.pose.position.x    = float(x)
        m.pose.position.y    = float(y)
        m.pose.position.z    = 0.3
        m.pose.orientation.w = 1.0
        m.scale.z            = 0.2
        m.color              = color
        m.text               = text
        m.lifetime           = Duration(sec=0)
        return m
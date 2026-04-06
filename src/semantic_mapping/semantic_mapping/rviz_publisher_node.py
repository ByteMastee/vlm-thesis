from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from builtin_interfaces.msg import Duration


class RvizPublisherNode:
    def __init__(self, logger):
        self.logger = logger

    # --- Final map markers (called once after clustering) ---
    def build_marker_array(self, object_stack, ground_truth, robot_path, clock):
        marker_array = MarkerArray()
        marker_id    = 0

        for label, data in object_stack.items():
            ox = data['x']
            oy = data['y']
            marker_array.markers.append(self._make_sphere_marker(
                marker_id, ox, oy, 'detected',
                ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0), clock
            ))
            marker_id += 1
            marker_array.markers.append(self._make_text_marker(
                marker_id, ox, oy, f'Det: {label}\n({ox:.2f},{oy:.2f})',
                'detected_labels', ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0), clock
            ))
            marker_id += 1

        for label, (gx, gy) in ground_truth.items():
            marker_array.markers.append(self._make_sphere_marker(
                marker_id, gx, gy, 'ground_truth',
                ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0), clock
            ))
            marker_id += 1
            marker_array.markers.append(self._make_text_marker(
                marker_id, gx, gy, f'GT: {label}\n({gx},{gy})',
                'gt_labels', ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0), clock
            ))
            marker_id += 1

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

        self.logger.info(f'Final marker array built — {len(marker_array.markers)} markers.')
        return marker_array

    # --- GT markers (called once at start) ---
    def build_gt_markers(self, ground_truth, clock):
        marker_array = MarkerArray()
        marker_id    = 0
        for label, (gx, gy) in ground_truth.items():
            marker_array.markers.append(self._make_sphere_marker(
                marker_id, gx, gy, 'ground_truth',
                ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0), clock
            ))
            marker_id += 1
            marker_array.markers.append(self._make_text_marker(
                marker_id, gx, gy, f'GT: {label}\n({gx},{gy})',
                'gt_labels', ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0), clock
            ))
            marker_id += 1
        return marker_array

    # --- Live markers per frame ---
    def build_live_markers(self, robot_x, robot_y, rays, candidates, clock):
        marker_array = MarkerArray()
        marker_id    = 0

        # Robot position sphere
        marker_array.markers.append(self._make_sphere_marker(
            marker_id, robot_x, robot_y, 'robot_position',
            ColorRGBA(r=0.0, g=1.0, b=1.0, a=1.0), clock,
            scale=0.2
        ))
        marker_id += 1

        # Rays — one arrow per detection
        for origin, ray, ray_length in rays:
            end = origin + ray * ray_length
            arrow                    = Marker()
            arrow.header.frame_id    = 'odom'
            arrow.header.stamp       = clock.now().to_msg()
            arrow.ns                 = 'live_rays'
            arrow.id                 = marker_id
            arrow.type               = Marker.ARROW
            arrow.action             = Marker.ADD
            arrow.scale.x            = 0.04
            arrow.scale.y            = 0.08
            arrow.scale.z            = 0.08
            arrow.color              = ColorRGBA(r=1.0, g=1.0, b=0.0, a=0.8)
            arrow.pose.orientation.w = 1.0
            arrow.lifetime           = Duration(sec=3)

            p_start   = Point()
            p_start.x = float(origin[0])
            p_start.y = float(origin[1])
            p_start.z = float(origin[2])

            p_end   = Point()
            p_end.x = float(end[0])
            p_end.y = float(end[1])
            p_end.z = float(end[2])

            arrow.points = [p_start, p_end]
            marker_array.markers.append(arrow)
            marker_id += 1

        # # Candidate points
        # for cx, cy in candidates:
        #     marker_array.markers.append(self._make_sphere_marker(
        #         marker_id, cx, cy, 'candidates',
        #         ColorRGBA(r=1.0, g=0.5, b=0.0, a=0.7), clock,
        #         scale=0.1
        #     ))
        #     marker_id += 1

        return marker_array

    # --- Private helpers ---

    def _make_sphere_marker(self, marker_id, x, y, ns, color, clock, scale=0.2):
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
        m.scale.x            = scale
        m.scale.y            = scale
        m.scale.z            = scale
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
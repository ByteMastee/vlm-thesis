#!/usr/bin/env python3
#
# exploration_launch.py
#
# Navigation in unknown environment:
#   - You give 2D Goal Pose in RViz on the explored (white) area
#   - Robot drives there avoiding obstacles using d455 + d435i cameras
#   - RTAB-Map builds the map as the robot moves
#   - Map is saved automatically every 30 seconds to ~/explored_maps/
#
# Launch order:
#   Step 1 — Gazebo + robot (your existing launch):
#       ros2 launch uvc1_gazebo uvc1_my_world_nav2_rtabmap_launch.py
#
#   Step 2 — This file:
#       ros2 launch uvc1_gazebo exploration_launch.py
#
#   Step 3 — In RViz:
#       Click "2D Goal Pose" → click on a WHITE area of the map
#       Robot navigates there, map grows as it drives
#       Repeat with new goals to explore more of the environment
#
#   Save map manually at any time:
#       ros2 run nav2_map_server map_saver_cli -f ~/explored_maps/my_map
#
# Authors: Pravin Oli  /  generated with Claude

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    # -----------------------------------------------------------------------
    # Package paths
    # -----------------------------------------------------------------------
    pkg_uvc1 = get_package_share_directory("uvc1_gazebo")
    pkg_nav2 = get_package_share_directory("nav2_bringup")

    # -----------------------------------------------------------------------
    # Launch arguments
    # -----------------------------------------------------------------------
    use_sim_time      = LaunchConfiguration("use_sim_time")
    map_save_dir      = LaunchConfiguration("map_save_dir")
    map_save_interval = LaunchConfiguration("map_save_interval")

    declare_use_sim_time = DeclareLaunchArgument(
        "use_sim_time",
        default_value="true",
        description="Use simulation clock",
    )
    declare_map_save_dir = DeclareLaunchArgument(
        "map_save_dir",
        default_value=os.path.expanduser("~/explored_maps"),
        description="Directory to save maps into",
    )
    declare_map_save_interval = DeclareLaunchArgument(
        "map_save_interval",
        default_value="30.0",
        description="Seconds between automatic map saves",
    )

    # -----------------------------------------------------------------------
    # 1. RGBD Sync — d455
    #    Combines color + depth → /rgbd_image0 for RTAB-Map
    #    Topic names match your Gazebo setup exactly
    # -----------------------------------------------------------------------
    rgbd_sync_d455 = Node(
        package="rtabmap_sync",
        executable="rgbd_sync",
        name="rgbd_sync_d455",
        output="screen",
        parameters=[{"use_sim_time": use_sim_time, "approx_sync": True}],
        remappings=[
            ("rgb/image",       "/d455/rgb/d455_rgb/image_raw"),
            ("rgb/camera_info", "/d455/rgb/d455_rgb/camera_info"),
            ("depth/image",     "/d455/depth/d455_depth/depth/image_raw"),
            ("rgbd_image",      "/rgbd_image0"),
        ],
    )

    # -----------------------------------------------------------------------
    # 2. RGBD Sync — d435i
    #    Combines color + depth → /rgbd_image1 for RTAB-Map
    # -----------------------------------------------------------------------
    rgbd_sync_d435i = Node(
        package="rtabmap_sync",
        executable="rgbd_sync",
        name="rgbd_sync_d435i",
        output="screen",
        parameters=[{"use_sim_time": use_sim_time, "approx_sync": True}],
        remappings=[
            ("rgb/image",       "/d435i/rgb/d435i_rgb/image_raw"),
            ("rgb/camera_info", "/d435i/rgb/d435i_rgb/camera_info"),
            ("depth/image",     "/d435i/depth/d435i_depth/depth/image_raw"),
            ("rgbd_image",      "/rgbd_image1"),
        ],
    )

    # -----------------------------------------------------------------------
    # 3. RTAB-Map SLAM
    #    Reads /rgbd_image0 + /rgbd_image1
    #    Publishes /map (OccupancyGrid) → Nav2 reads this for path planning
    #    Publishes /odom → Nav2 uses this for localization
    # -----------------------------------------------------------------------
    rtabmap_node = Node(
        package="rtabmap_slam",
        executable="rtabmap",
        name="rtabmap",
        output="screen",
        parameters=[
            os.path.join(pkg_uvc1, "config", "rtabmap_params_exploration.yaml"),
        ],
        remappings=[
            ("rgbd_image0", "/rgbd_image0"),
            ("rgbd_image1", "/rgbd_image1"),
        ],
    )

    # -----------------------------------------------------------------------
    # 4. Nav2 — delayed 5s so RTAB-Map publishes /map first
    #
    #    Uses your existing nav2_params.yaml — no changes needed.
    #    Your costmaps already use d455 + d435i point clouds for obstacles.
    #    allow_unknown: true in planner means Nav2 can plan into gray areas.
    # -----------------------------------------------------------------------
    nav2_launch = TimerAction(
        period=5.0,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(pkg_nav2, "launch", "navigation_launch.py")
                ),
                launch_arguments={
                    "use_sim_time": use_sim_time,
                    "params_file":  os.path.join(
                        pkg_uvc1, "config", "nav2_params.yaml"
                    ),
                    "autostart": "true",
                }.items(),
            )
        ],
    )

    # -----------------------------------------------------------------------
    # 5. Periodic map saver — delayed 20s so Nav2 + RTAB-Map are stable
    #    Saves ~/explored_maps/map_YYYYMMDD_HHMMSS.pgm + .yaml every 30s
    # -----------------------------------------------------------------------
    map_saver_node = TimerAction(
        period=20.0,
        actions=[
            Node(
                package="uvc1_gazebo",
                executable="map_saver_periodic",
                name="map_saver_periodic",
                output="screen",
                parameters=[
                    {
                        "use_sim_time":      use_sim_time,
                        "map_save_dir":      map_save_dir,
                        "map_save_interval": map_save_interval,
                    }
                ],
            )
        ],
    )

    # -----------------------------------------------------------------------
    # Assemble
    # -----------------------------------------------------------------------
    ld = LaunchDescription()

    ld.add_action(declare_use_sim_time)
    ld.add_action(declare_map_save_dir)
    ld.add_action(declare_map_save_interval)

    # t=0s  — RTAB-Map starts building the map immediately
    ld.add_action(rgbd_sync_d455)
    ld.add_action(rgbd_sync_d435i)
    ld.add_action(rtabmap_node)

    # t=5s  — Nav2 starts (map already exists by now)
    ld.add_action(nav2_launch)

    # t=20s — map saver starts (everything stable by now)
    ld.add_action(map_saver_node)

    return ld

#!/usr/bin/env python3
# =============================================================================
#  bringup_launch_amcl.py  —  uvc1 Virofighter Robot
#
#  Standard Nav2 bringup with AMCL localization.
#  NO RTAB-Map dependency — uses depthimage_to_laserscan to generate /scan
#  from the D435i depth camera, then feeds it to AMCL for localization.
#
#  This file replaces bringup_launch_multicamera.py for localization mode.
#  For SLAM, you still need RTAB-Map (or slam_toolbox with laser scan).
#
#  USAGE:
#    # Localization in simulation (default):
#    ros2 launch uvc1_gazebo bringup_launch_amcl_d435i.py map_name:=house1
#
#    # With explicit sim time:
#    ros2 launch uvc1_gazebo bringup_launch_amcl_d435i.py map_name:=house1 use_sim_time:=true
#
#    # Real robot:
#    ros2 launch uvc1_gazebo bringup_launch_amcl_d435i.py map_name:=house1 use_sim_time:=false
#
#  WHAT IT LAUNCHES:
#    1. depthimage_to_laserscan (D435i depth → /scan)
#    2. Nav2 bringup_launch.py  (map_server + AMCL + navigation stack)
#
#  PREREQUISITES:
#    sudo apt install ros-humble-depthimage-to-laserscan
#    # Your Gazebo world + robot must already be running (spawned separately)
# =============================================================================


# =============================================================================
#  bringup_launch_amcl.py  —  uvc1 Virofighter Robot
#
#  Standard Nav2 bringup with AMCL localization.
#  Uses depthimage_to_laserscan to generate /scan from D435i depth camera.
#
#  FIX: The Gazebo depth camera plugin publishes camera_info with
#  frame_id "camera_depth_frame", and depthimage_to_laserscan uses that
#  frame for the output /scan regardless of output_frame_id parameter.
#  But our URDF TF tree has "camera_d435i_depth_optical_frame".
#  Solution: publish a static identity transform linking them.
#
#  USAGE:
#    Terminal 1: ros2 launch uvc1_gazebo uvc1_my_world_nav2_rtabmap.launch.py
#    Terminal 2: ros2 launch uvc1_gazebo bringup_launch_amcl_d435i.py map_name:=house1
# =============================================================================

import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def setup_map_path(context, *args, **kwargs):
    """Resolve map_name → full path to .yaml map file at launch time."""
    pkg = get_package_share_directory("uvc1_gazebo")
    map_name = LaunchConfiguration("map_name").perform(context)
    use_sim_time = LaunchConfiguration("use_sim_time")

    map_dir = Path(pkg) / "maps" / map_name
    if not map_dir.exists():
        raise RuntimeError(f"Map folder not found: {map_dir}")

    yaml_path = str(map_dir / f"{map_name}.yaml")
    params_file = os.path.join(pkg, "config", "nav2_params.yaml")
    pkg_nav2 = get_package_share_directory("nav2_bringup")

    # Standard Nav2 bringup — includes map_server, AMCL, and full navigation stack
    # Delayed 5s to let Gazebo + robot_state_publisher settle TF tree
    nav2_bringup = TimerAction(
        period=3.0,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(pkg_nav2, "launch", "bringup_launch.py")
                ),
                launch_arguments={
                    "use_sim_time": use_sim_time,
                    "map": yaml_path,
                    "params_file": params_file,
                    "autostart": "true",
                }.items(),
            )
        ],
    )

    return [nav2_bringup]


def generate_launch_description():
    # --- Arguments ---
    declare_use_sim_time = DeclareLaunchArgument(
        "use_sim_time",
        default_value="true",
        description="Use simulation clock",
    )

    declare_map_name = DeclareLaunchArgument(
        "map_name",
        default_value="house1",
        description="Map folder name inside uvc1_gazebo/maps/",
    )

    # =========================================================================
    # STATIC TF: camera_d435i_depth_optical_frame → camera_depth_frame
    #
    # The Gazebo depth plugin publishes camera_info with frame_id
    # "camera_depth_frame". depthimage_to_laserscan then publishes /scan
    # with that same frame. But our URDF only has
    # "camera_d435i_depth_optical_frame" in the TF tree.
    #
    # This identity transform (all zeros) tells TF they are the same frame.
    # Parent: camera_d435i_depth_optical_frame (exists in URDF TF tree)
    # Child:  camera_depth_frame (used by Gazebo / depthimage_to_laserscan)
    # =========================================================================
    static_tf_camera_depth = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="static_tf_camera_depth_frame",
        arguments=[
            "--x",
            "0",
            "--y",
            "0",
            "--z",
            "0",
            "--roll",
            "0",
            "--pitch",
            "0",
            "--yaw",
            "0",
            "--frame-id",
            "camera_d435i_depth_optical_frame",
            "--child-frame-id",
            "camera_depth_frame",
        ],
        parameters=[{"use_sim_time": LaunchConfiguration("use_sim_time")}],
    )

    # =========================================================================
    # depthimage_to_laserscan  (D435i depth → /scan)
    #
    # Verified topic names from `ros2 topic list`:
    #   /d435i/depth/d435i_depth/depth/image_raw
    #   /d435i/depth/d435i_depth/depth/camera_info
    # =========================================================================
    depth_to_scan = Node(
        package="depthimage_to_laserscan",
        executable="depthimage_to_laserscan_node",
        name="depthimage_to_laserscan",
        output="screen",
        parameters=[
            {
                "use_sim_time": LaunchConfiguration("use_sim_time"),
                # How many pixel rows (centered) to collapse into 1 scan line
                "scan_height": 10,
                # Scan timing (match camera update rate = 30 Hz)
                "scan_time": 0.033,
                # Range limits — match D435i depth clip in URDF
                "range_min": 0.6,  # D435i depth clip near = 0.6m
                "range_max": 6.0,  # D435i depth clip far = 6.0m
                # Output frame — the depth optical frame from URDF
                "output_frame_id": "camera_d435i_depth_optical_frame",
            }
        ],
        remappings=[
            ("depth", "/d435i/depth/d435i_depth/depth/image_raw"),
            ("depth_camera_info", "/d435i/depth/d435i_depth/depth/camera_info"),
            ("scan", "/scan"),
        ],
    )

    # --- Build launch description ---
    ld = LaunchDescription()
    ld.add_action(declare_use_sim_time)
    ld.add_action(declare_map_name)

    # Static TF bridge — must start before anything reads /scan
    ld.add_action(static_tf_camera_depth)

    # Depth-to-scan converter
    ld.add_action(depth_to_scan)

    # Nav2 bringup (delayed 10s)
    ld.add_action(OpaqueFunction(function=setup_map_path))

    return ld


"""
QUICK START:

1. Install dependency:
   sudo apt install ros-humble-depthimage-to-laserscan

2. Terminal 1 — Start Gazebo + robot:
   ros2 launch uvc1_gazebo uvc1_my_world_nav2_rtabmap.launch.py

3. Terminal 2 — Start Nav2 with AMCL:
   ros2 launch uvc1_gazebo bringup_launch_amcl_d435i.py map_name:=house1

4. In RViz:
   - Set initial pose with "2D Pose Estimate"
   - Send goals with "Nav2 Goal"

VERIFY /scan is working:
   ros2 topic echo /scan --once
   ros2 topic hz /scan
"""

"""
USAGE:
  Terminal 1: ros2 launch uvc1_gazebo uvc1_my_world_nav2_rtabmap.launch.py
  Terminal 2: ros2 launch uvc1_gazebo bringup_launch_amcl_d435i.py map_name:=house1
  Then in RViz: use "2D Pose Estimate" to set initial position

VERIFY:
  ros2 topic hz /scan                          # ~10-30 Hz
  ros2 run tf2_ros tf2_echo map base_link      # should show transform after initial pose
  ros2 topic echo /amcl_pose --once            # should show localized pose
"""

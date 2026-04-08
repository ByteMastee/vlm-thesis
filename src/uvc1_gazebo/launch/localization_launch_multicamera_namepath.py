#!/usr/bin/env python3

import os
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory("uvc1_gazebo")

    use_sim_time = LaunchConfiguration("use_sim_time")
    map_yaml_file = LaunchConfiguration("map")
    db_file = LaunchConfiguration("database")

    # -------------------
    # ENV
    # -------------------
    stdout_linebuf_envvar = SetEnvironmentVariable(
        "RCUTILS_LOGGING_BUFFERED_STREAM", "1"
    )

    # -------------------
    # ARGUMENTS
    # -------------------
    declare_use_sim_time = DeclareLaunchArgument("use_sim_time", default_value="true")

    declare_map = DeclareLaunchArgument(
        "map",
        default_value=os.path.join(pkg, "maps", "house1", "house1.yaml"),
    )

    declare_db = DeclareLaunchArgument(
        "database",
        default_value=os.path.join(pkg, "maps", "house1", "house1.db"),
    )

    # -------------------
    # MAP SERVER (Nav2)
    # -------------------
    map_server = Node(
        package="nav2_map_server",
        executable="map_server",
        name="map_server",
        output="screen",
        parameters=[{"use_sim_time": use_sim_time, "yaml_filename": map_yaml_file}],
    )

    lifecycle_manager = Node(
        package="nav2_lifecycle_manager",
        executable="lifecycle_manager",
        name="lifecycle_manager_localization",
        output="screen",
        parameters=[
            {
                "use_sim_time": use_sim_time,
                "autostart": True,
                "node_names": ["map_server"],
            }
        ],
    )

    # -------------------
    # RGBD SYNC (MULTICAMERA)
    # -------------------
    rgbd_sync_d455 = Node(
        package="rtabmap_sync",
        executable="rgbd_sync",
        name="rgbd_sync_d455",
        output="screen",
        parameters=[{"use_sim_time": use_sim_time, "approx_sync": True}],
        remappings=[
            ("rgb/image", "/d455/rgb/d455_rgb/image_raw"),
            ("rgb/camera_info", "/d455/rgb/d455_rgb/camera_info"),
            ("depth/image", "/d455/depth/d455_depth/depth/image_raw"),
            ("rgbd_image", "/rgbd_image0"),
        ],
    )

    rgbd_sync_d435i = Node(
        package="rtabmap_sync",
        executable="rgbd_sync",
        name="rgbd_sync_d435i",
        output="screen",
        parameters=[{"use_sim_time": use_sim_time, "approx_sync": True}],
        remappings=[
            ("rgb/image", "/d435i/rgb/d435i_rgb/image_raw"),
            ("rgb/camera_info", "/d435i/rgb/d435i_rgb/camera_info"),
            ("depth/image", "/d435i/depth/d435i_depth/depth/image_raw"),
            ("rgbd_image", "/rgbd_image1"),
        ],
    )

    # -------------------
    # RTABMAP LOCALIZATION (KEY PART)
    # -------------------
    rtabmap = Node(
        package="rtabmap_slam",
        executable="rtabmap",
        name="rtabmap",
        output="screen",
        parameters=[
            {
                "use_sim_time": use_sim_time,
                # LOAD YOUR SAVED MAP DATABASE
                "database_path": db_file,
                # LOCALIZATION MODE
                "Mem/IncrementalMemory": "false",
                "Mem/InitWMWithAllNodes": "true",
                # CRITICAL FIX
                "RGBD/StartAtOrigin": "true",
                "Reg/Strategy": "1",  # remove this for default PnP, 1=Icp+Feature2d, 2=Icp, 3=Feature2d
                # "Icp/MaxCorrespondenceDistance": "0.1", # 10cm is a good start for indoors, increase if you have fast motion or noisy depth
                # Sensors
                "subscribe_rgbd": True,
                "rgbd_cameras": 2,
                # TF setup
                "frame_id": "base_link",
                "Odom/FrameId": "odom",
                "Map/FrameId": "map",
            }
        ],
        remappings=[
            ("rgbd_image0", "/rgbd_image0"),
            ("rgbd_image1", "/rgbd_image1"),
            # allow RViz pose estimate
            ("/initialpose", "/rtabmap/initial_pose"),
            # prevent map conflict
            ("map", "/slam_map"),
        ],
    )

    # -------------------
    # BUILD LAUNCH
    # -------------------
    ld = LaunchDescription()

    ld.add_action(stdout_linebuf_envvar)

    ld.add_action(declare_use_sim_time)
    ld.add_action(declare_map)
    ld.add_action(declare_db)

    ld.add_action(map_server)
    ld.add_action(lifecycle_manager)

    ld.add_action(rgbd_sync_d455)
    ld.add_action(rgbd_sync_d435i)

    ld.add_action(rtabmap)

    return ld


"""
ros2 launch uvc1_gazebo localization_launch_multicamera_namepath.py map:=/home/pravin/vf_robot_model_ros2/src/uvc1_gazebo/maps/house1/house1.yaml database:=/home/pravin/vf_robot_model_ros2/src/uvc1_gazebo/maps/house1/house1.db
"""

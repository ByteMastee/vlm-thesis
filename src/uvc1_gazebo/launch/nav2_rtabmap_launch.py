#!/usr/bin/env python3
#
# Copyright  EUROKNOWS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
#
# Authors: Pravin Oli
# https://www.euroknows.com/en/home/

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import SetEnvironmentVariable, GroupAction, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from launch_ros.actions import Node, PushRosNamespace
from nav2_common.launch import RewrittenYaml


def generate_launch_description():

    # --- Directories ---
    pkg_uvc1 = get_package_share_directory("uvc1_gazebo")
    pkg_nav2 = get_package_share_directory("nav2_bringup")

    nav2_params_file = os.path.join(pkg_uvc1, "config", "nav2_params.yaml")

    # --- Launch configuration ---
    namespace = LaunchConfiguration("namespace", default="")
    use_namespace = LaunchConfiguration("use_namespace", default="false")
    use_sim_time = LaunchConfiguration("use_sim_time", default="true")
    autostart = LaunchConfiguration("autostart", default="true")
    log_level = LaunchConfiguration("log_level", default="info")

    remappings = [("/tf", "tf"), ("/tf_static", "tf_static")]

    # --- Rewritten YAML ---
    configured_nav2_params = RewrittenYaml(
        source_file=nav2_params_file,
        root_key=namespace,
        param_rewrites={"use_sim_time": use_sim_time},
        convert_types=True,
    )

    stdout_linebuf_envvar = SetEnvironmentVariable(
        "RCUTILS_LOGGING_BUFFERED_STREAM", "1"
    )

    # --- RGBD Sync Node for D455 (used for SLAM) ---
    rgbd_sync_d455 = Node(
        package="rtabmap_sync",
        executable="rgbd_sync",
        name="rgbd_sync_d455",
        output="screen",
        parameters=[
            {"use_sim_time": use_sim_time},
            {"approx_sync": True},
        ],
        remappings=[
            ("rgb/image", "/d455/rgb/d455_rgb/image_raw"),
            ("rgb/camera_info", "/d455/rgb/d455_rgb/camera_info"),
            ("depth/image", "/d455/depth/d455_depth/depth/image_raw"),
            ("rgbd_image", "/rgbd_image_d455"),
        ],
    )

    # --- RTAB-Map SLAM Node ---
    # rtabmap_slam_node = Node(
    #     package="rtabmap_ros",
    #     executable="rtabmap",
    #     name="rtabmap",
    #     output="screen",
    #     parameters=[
    #         configured_nav2_params
    #     ],  # all RTAB-Map params come from nav2_params.yaml
    #     remappings=[
    #         ("rgbd_image", "/rgbd_image_d455"),  # feed only D455
    #     ],
    # )

    rtabmap_slam_node = Node(
        package="rtabmap_slam",
        executable="rtabmap",
        name="rtabmap",
        output="screen",
        parameters=[configured_nav2_params],
        remappings=[
            ("rgbd_image", "/rgbd_image_d455"),
        ],
    )

    # --- Nav2 group ---
    nav2_group = GroupAction(
        [
            PushRosNamespace(condition=IfCondition(use_namespace), namespace=namespace),
            Node(
                package="rclcpp_components",
                executable="component_container_isolated",
                name="nav2_container",
                parameters=[configured_nav2_params, {"autostart": autostart}],
                arguments=["--ros-args", "--log-level", log_level],
                remappings=remappings,
                output="screen",
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(pkg_nav2, "launch", "navigation_launch.py")
                ),
                launch_arguments={
                    "namespace": namespace,
                    "use_sim_time": use_sim_time,
                    "autostart": autostart,
                    "params_file": nav2_params_file,
                    "use_composition": "True",
                    "use_respawn": "False",
                    "container_name": "nav2_container",
                }.items(),
            ),
        ]
    )

    # --- LaunchDescription ---
    ld = LaunchDescription()
    ld.add_action(stdout_linebuf_envvar)
    ld.add_action(rgbd_sync_d455)
    ld.add_action(rtabmap_slam_node)
    ld.add_action(nav2_group)

    return ld

#!/usr/bin/env python3
#
# Copyright  EUROKNOWS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Pravin Oli
# https://www.euroknows.com/en/home/

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.actions import Node


def generate_launch_description():
    # Directories
    pkg_gazebo_ros = get_package_share_directory("gazebo_ros")
    pkg_uvc1 = get_package_share_directory("uvc1_gazebo")
    launch_dir = os.path.join(pkg_uvc1, "launch")
    urdf_file = os.path.join(pkg_uvc1, "urdf", "uvc1_virofighter.urdf")

    # Launch configurations
    use_sim_time = LaunchConfiguration("use_sim_time", default="true")
    x_pose = LaunchConfiguration("x_pose", default="0.0")
    y_pose = LaunchConfiguration("y_pose", default="0.0")
    z_pose = LaunchConfiguration("z_pose", default="0.0")

    # World file
    world_file = os.path.join(pkg_uvc1, "worlds", "empty_world.world")

    # Start Gazebo server
    gzserver_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, "launch", "gzserver.launch.py")
        ),
        launch_arguments={"world": world_file}.items(),
    )

    # Start Gazebo client
    gzclient_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, "launch", "gzclient.launch.py")
        )
    )

    # Publish robot_state using URDF
    robot_state_publisher_cmd = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[
            {
                "use_sim_time": use_sim_time,
                "robot_description": Command(
                    ["xacro ", urdf_file, " use_sim_time:=", use_sim_time]
                ),
            }
        ],
    )

    # Spawn robot from URDF (Gazebo reads all <gazebo> tags inside links)
    spawn_robot_cmd = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        arguments=[
            "-entity",
            "uvc1_virofighter",
            "-x",
            x_pose,
            "-y",
            y_pose,
            "-z",
            z_pose,
            "-topic",
            "robot_description",  # uses URDF published by robot_state_publisher
        ],
        output="screen",
    )

    # Optional GUI teleop node
    gui_teleop_node = Node(
        package="rqt_robot_steering",
        executable="rqt_robot_steering",
        name="rqt_robot_steering",
        output="screen",
        parameters=[{"use_sim_time": use_sim_time}],
    )

    # LaunchDescription
    ld = LaunchDescription()
    ld.add_action(gzserver_cmd)
    ld.add_action(gzclient_cmd)
    ld.add_action(robot_state_publisher_cmd)
    ld.add_action(spawn_robot_cmd)
    ld.add_action(gui_teleop_node)

    return ld

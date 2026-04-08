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
# Authors: Pravin Oli
# https://www.euroknows.com/en/home/

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    UVC1_MODEL = os.environ.get("UVC1_MODEL", "virofighter")

    # Launch configuration variables specific to simulation
    x_pose = LaunchConfiguration("x_pose", default="0.0")
    y_pose = LaunchConfiguration("y_pose", default="0.0")
    theta  = LaunchConfiguration("theta",  default="0.0")

    # Declare the launch arguments
    declare_x_position_cmd = DeclareLaunchArgument(
        "x_pose", default_value="0.0", description="Initial X position of the robot"
    )
    declare_y_position_cmd = DeclareLaunchArgument(
        "y_pose", default_value="0.0", description="Initial Y position of the robot"
    )
    declare_theta_cmd = DeclareLaunchArgument(
        "theta", default_value="0.0", description="Initial yaw (radians) of the robot"
    )

    # Plan A: spawn from /robot_description topic published by robot_state_publisher.
    # This means the URDF (with its <gazebo> plugin tags) is the single source of
    # truth — model.sdf is no longer needed and can be removed.
    start_gazebo_ros_spawner_cmd = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        arguments=[
            "-entity", UVC1_MODEL,
            "-topic", "/robot_description",   # ← reads the URDF from RSP
            "-x", x_pose,
            "-y", y_pose,
            "-z", "0.01",
            "-Y", theta,
        ],
        output="screen",
    )

    ld = LaunchDescription()
    ld.add_action(declare_x_position_cmd)
    ld.add_action(declare_y_position_cmd)
    ld.add_action(declare_theta_cmd)
    ld.add_action(start_gazebo_ros_spawner_cmd)

    return ld

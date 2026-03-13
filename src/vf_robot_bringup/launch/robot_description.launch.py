from launch import LaunchDescription
from launch.substitutions import Command, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():

    # =====================================================
    # Robot description (XACRO → URDF string)
    # =====================================================
    robot_description = ParameterValue(
        Command(
            [
                "xacro ",
                PathJoinSubstitution(
                    [
                        FindPackageShare("vf_robot_description"),
                        "urdf",
                        "robot_urdf.xacro",
                    ]
                ),
            ]
        ),
        value_type=str,
    )

    # =====================================================
    # RViz config
    # =====================================================
    rviz_config = PathJoinSubstitution(
        [FindPackageShare("vf_robot_bringup"), "rviz", "default.rviz"]
    )

    # =====================================================
    # Launch description
    # =====================================================
    return LaunchDescription(
        [
            # Publishes joint states (GUI sliders)
            Node(
                package="joint_state_publisher_gui",
                executable="joint_state_publisher_gui",
                output="screen",
            ),
            # Publishes TF using robot_description
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                output="screen",
                parameters=[{"robot_description": robot_description}],
            ),
            # RViz visualization
            Node(
                package="rviz2",
                executable="rviz2",
                arguments=["-d", rviz_config],
                output="screen",
            ),
        ]
    )

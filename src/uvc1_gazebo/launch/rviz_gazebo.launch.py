import os

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    vf_robot_model_dir = get_package_share_directory("vf_robot_model")
    gazebo_ros_dir = get_package_share_directory("gazebo_ros")

    # ------------------------------------------------
    # Paths
    # ------------------------------------------------

    urdf_file = os.path.join(
        vf_robot_model_dir, "models", "UVC_robot_new", "urdf", "robot.urdf"
    )

    world_file = os.path.join(vf_robot_model_dir, "worlds", "hospital.world")

    rviz_config = os.path.join(vf_robot_model_dir, "rviz", "default.rviz")

    # ------------------------------------------------
    # Read URDF
    # ------------------------------------------------

    with open(urdf_file, "r") as f:
        robot_urdf = f.read()

    # ------------------------------------------------
    # Gazebo model path
    # ------------------------------------------------

    set_model_path = SetEnvironmentVariable(
        name="GAZEBO_MODEL_PATH",
        value=os.path.join(vf_robot_model_dir, "models"),
    )

    # ------------------------------------------------
    # Gazebo
    # ------------------------------------------------

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_ros_dir, "launch", "gazebo.launch.py")
        ),
        launch_arguments={"world": world_file}.items(),
    )

    # ------------------------------------------------
    # Robot State Publisher (TF from joint_states)
    # ------------------------------------------------

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[{"robot_description": robot_urdf}],
        output="screen",
    )

    # ------------------------------------------------
    # Spawn robot into Gazebo
    # ------------------------------------------------

    spawn_robot = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        arguments=["-topic", "robot_description", "-entity", "robot"],
        output="screen",
    )

    # ------------------------------------------------
    # RViz
    # ------------------------------------------------

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        arguments=["-d", rviz_config],
        output="screen",
    )

    # ------------------------------------------------
    # Optional teleop
    # ------------------------------------------------

    gui_teleop_node = Node(
        package="rqt_robot_steering",
        executable="rqt_robot_steering",
        output="screen",
        parameters=[{"cmd_vel_topic": "/cmd_vel"}],
    )

    # ------------------------------------------------
    # Launch!
    # ------------------------------------------------

    return LaunchDescription(
        [
            set_model_path,
            gazebo,
            robot_state_publisher,
            spawn_robot,
            rviz,
            gui_teleop_node,
        ]
    )

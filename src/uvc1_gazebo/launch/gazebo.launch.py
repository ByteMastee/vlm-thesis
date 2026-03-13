import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Get package directories
    vf_robot_model_dir = get_package_share_directory("vf_robot_model")
    gazebo_ros_dir = get_package_share_directory("gazebo_ros")

    # Set Gazebo model path
    model_path = os.path.join(vf_robot_model_dir, "models")
    set_model_path = SetEnvironmentVariable("GAZEBO_MODEL_PATH", model_path)

    # Path to robot URDF
    urdf_file = os.path.join(
        vf_robot_model_dir, "models", "UVC_robot_new", "urdf", "robot.urdf"
    )

    # Read URDF content
    with open(urdf_file, "r") as f:
        robot_urdf = f.read()

    # Choose world file
    world_file = os.path.join(vf_robot_model_dir, "worlds", "hospital.world")

    # Launch Gazebo
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_ros_dir, "launch", "gazebo.launch.py")
        ),
        launch_arguments={"world": world_file}.items(),
    )

    # Robot state publisher
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[{"robot_description": robot_urdf}],
        output="screen",
    )

    # Spawn robot in Gazebo
    spawn_robot = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        arguments=[
            "-topic",
            "robot_description",  # this matches robot_state_publisher
            "-entity",
            "robot",
        ],
        output="screen",
    )

    # GUI Teleop node (rqt_robot_steering)
    gui_teleop_node = Node(
        package="rqt_robot_steering",
        executable="rqt_robot_steering",
        name="rqt_robot_steering",
        output="screen",
        parameters=[
            {"cmd_vel_topic": "/cmd_vel"}
        ],  # make sure it publishes to robot's cmd_vel
    )

    return LaunchDescription(
        [
            set_model_path,
            robot_state_publisher,  # <-- first
            gazebo,
            spawn_robot,  # <-- second
            gui_teleop_node,  # <-- added GUI teleop
        ]
    )

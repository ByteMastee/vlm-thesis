from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='depth_image_proc',
            executable='point_cloud_xyz_radial_node',
            name='d455_depth_to_cloud',
            output='screen',
            parameters=[
                {'use_sim_time': True},
                {'queue_size': 30},
            ],
            remappings=[
                ('depth/image_raw', '/d455/depth/d455_depth/depth/image_raw'),
                ('depth/camera_info', '/d455/depth/d455_depth/camera_info'),
                ('points', '/thesis_mapping/d455_points'),
            ]
        )
    ])
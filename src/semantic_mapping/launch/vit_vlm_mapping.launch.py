from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import os

RUN_NAME = 'VITp2_3'
BASE_OUTPUT_DIR = '/root/UVC_ws/vf_robot_model_ros2/Final_Output/RW_FinalOutput'
OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, RUN_NAME)


def generate_launch_description():

    fisheye_rectify_node = Node(
        package='semantic_mapping',
        executable='fisheye_rectify_node',
        name='fisheye_rectify_node',
        output='screen',
    )

    ros_node_vit = Node(
        package='semantic_mapping',
        executable='ros_node_vit',
        name='ros_node_vit',
        output='screen',
        parameters=[{
            'run_name':                 RUN_NAME,
            'output_dir':               OUTPUT_DIR,
            'image_topic':              '/fisheye_front/fisheye_front/image_rect',
            'cam_info_topic':           '/fisheye_front/fisheye_front/camera_info',
            'odom_topic':               '/odom',
            'frame_skip':               15,
            'min_angle_deg':            10.0,
            'dbscan_eps':               0.8,
            'dbscan_min_samples':       4,
            'ray_length':               8.0,
            'process_delay':            100.0,
            'env_frame_interval':       5,
            'sam2_checkpoint':          '/root/sam2_checkpoints/sam2.1_hiera_small.pt',
            'sam2_model_cfg':           'configs/sam2.1/sam2.1_hiera_s.yaml',
            'points_per_side':          16,
            'pred_iou_thresh':          0.85,
            'stability_score_thresh':   0.88,
            'min_mask_region_area':     2500,
            'max_mask_area_fraction':   0.10,
            'max_regions':              6,
        }]
    )

    vlm_label_node_vit = TimerAction(
        period=5.0,
        actions=[
            Node(
                package='semantic_mapping',
                executable='vlm_label_node_vit',
                name='vlm_label_node_vit',
                output='screen',
                parameters=[{
                    'run_name':           RUN_NAME,
                    'output_dir':         OUTPUT_DIR,
                    'model_path':         '/root/UVC_ws/models/qwen2.5-vl-3b',
                    'max_new_tokens':     128,
                    'env_sample_count':   5,
                    'min_angle_deg':      8.0,
                    'dbscan_eps':         0.8,
                    'dbscan_min_samples': 4,
                }]
            )
        ]
    )

    return LaunchDescription([
        fisheye_rectify_node,
        ros_node_vit,
        vlm_label_node_vit,
    ])
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    run_name_arg = DeclareLaunchArgument(
        'run_name', default_value='run_vit_03',
        description='Run identifier for output files'
    )
    process_delay_arg = DeclareLaunchArgument(
        'process_delay', default_value='400.0',
        description='Seconds to wait before triggering final processing'
    )
    output_dir_arg = DeclareLaunchArgument(
        'output_dir', default_value='',
        description='Override output directory (leave empty to auto-build from run_name)'
    )

    run_name      = LaunchConfiguration('run_name')
    process_delay = LaunchConfiguration('process_delay')
    output_dir    = LaunchConfiguration('output_dir')

    ros_node_vit = Node(
        package='semantic_mapping',
        executable='ros_node_vit',
        name='ros_node_vit',
        output='screen',
        parameters=[{
            'run_name':                 run_name,
            'process_delay':            process_delay,
            'output_dir':               output_dir,
            'image_topic':              '/fisheye_front/fisheye_front/image_raw',
            'cam_info_topic':           '/fisheye_front/fisheye_front/camera_info',
            'odom_topic':               '/odom',
            'frame_skip':               12,
            'min_angle_deg':            8.0,
            'dbscan_eps':               1.0,
            'dbscan_min_samples':       3,
            'ray_length':               8.0,
            'env_frame_interval':       20,
            'sam2_checkpoint':          '/root/sam2_checkpoints/sam2.1_hiera_small.pt',
            'sam2_model_cfg':           'configs/sam2.1/sam2.1_hiera_s.yaml',
            'points_per_side':          16,
            'pred_iou_thresh':          0.82,
            'stability_score_thresh':   0.85,
            'min_mask_region_area':     1500,
            'max_mask_area_fraction':   0.20,
            'max_regions':              12,
            'ground_truth':             [
                'chair_1:-3.0:2.0',
                'chair_2:-3.5:-2.5',
                'couch:3.5:0.0',
                'table:2.0:2.5'
            ],
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
                    'run_name':         run_name,
                    'output_dir':       output_dir,
                    'model_path':       '/root/UVC_ws/models/qwen2.5-vl-3b',
                    'max_new_tokens':   128,
                    'env_sample_count': 5,
                }]
            )
        ]
    )

    return LaunchDescription([
        run_name_arg,
        process_delay_arg,
        output_dir_arg,
        ros_node_vit,
        vlm_label_node_vit,
    ])
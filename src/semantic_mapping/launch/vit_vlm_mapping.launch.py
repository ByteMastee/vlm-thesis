from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    run_name_arg = DeclareLaunchArgument(
        'run_name', default_value='Env5_Path2',
        description='Run identifier for output files'
    )
    process_delay_arg = DeclareLaunchArgument(
        'process_delay', default_value='100.0',
        description='Seconds to wait before triggering final processing'
    )
    output_dir_arg = DeclareLaunchArgument(
        'output_dir', default_value='/root/UVC_ws/vf_robot_model_ros2/Final_Output/ViT/Env5_Path2',
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
            'frame_skip':               20,
            'min_angle_deg':            5.0,
            'dbscan_eps':               1.5,
            'dbscan_min_samples':       2,
            'ray_length':               8.0,
            'env_frame_interval':       5,
            'sam2_checkpoint':          '/root/sam2_checkpoints/sam2.1_hiera_small.pt',
            'sam2_model_cfg':           'configs/sam2.1/sam2.1_hiera_s.yaml',
            'points_per_side':          8,
            'pred_iou_thresh':          0.90,
            'stability_score_thresh':   0.92,
            'min_mask_region_area':     2000,
            'max_mask_area_fraction':   0.10,
            'max_regions':              8,
            'ground_truth':             [
                'reception_table:-5.0:5.5',
                'reception_chair:-5.0:6.6',
                'dustbin:0.0:6.0',
                'chair_1:4.5:5.5',
                'chair_2:5.7:5.5',
                'chair_3:6.9:5.5',
                'chair_4:8.1:5.5',
                'potted_plant:6.0:-4.5',
                'office_desk:7.5:2.0',
                'office_chair:7.5:0.9',
                'bookshelf:9.7:0.5',
                'filing_cabinet:4.2:-1.8',
                'operation_table:-6.5:0.5',
                'instrument_trolley:-5.5:-0.5',
                'medical_monitor:-7.8:1.5',
                'iv_stand:-7.5:-0.5'
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
                    'run_name':           run_name,
                    'output_dir':         output_dir,
                    'model_path':         '/root/UVC_ws/models/qwen2.5-vl-3b',
                    'max_new_tokens':     128,
                    'env_sample_count':   5,
                    'min_angle_deg':      2.0,
                    'dbscan_eps':         1.0,
                    'dbscan_min_samples': 2,
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
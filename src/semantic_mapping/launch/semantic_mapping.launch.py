import os
from launch import LaunchDescription
from launch_ros.actions import Node

RUN_NAME        = 'LS_01'
BASE_OUTPUT_DIR = '/root/UVC_ws/vf_robot_model_ros2/Final_Output/LiveStream_Yolo'
OUTPUT_DIR      = os.path.join(BASE_OUTPUT_DIR, RUN_NAME)


def generate_launch_description():
    return LaunchDescription([

        # --- Fisheye Rectification Node ---
        Node(
            package    = 'semantic_mapping',
            executable = 'fisheye_rectify_node',
            name       = 'fisheye_rectify_node',
            output     = 'screen',
        ),

        # --- VLM Label Node ---
        Node(
            package    = 'semantic_mapping',
            executable = 'vlm_label_node',
            name       = 'vlm_label_node',
            output     = 'screen',
            parameters = [
                {'run_name'         : RUN_NAME},
                {'output_dir'       : OUTPUT_DIR},
                {'model_path'       : '/root/UVC_ws/models/qwen2.5-vl-3b'},
                {'max_new_tokens'   : 128},
                {'env_sample_count' : 4},
            ]
        ),

        # --- ROS Bridge Node ---
        Node(
            package    = 'semantic_mapping',
            executable = 'ros_node',
            name       = 'ros_node',
            output     = 'screen',
            parameters = [
                {'run_name'          : RUN_NAME},
                {'image_topic'       : '/fisheye_front/fisheye_front/image_rect'},
                {'cam_info_topic'    : '/fisheye_front/fisheye_front/camera_info'},
                {'odom_topic'        : '/odom'},
                {'frame_skip'        : 10},
                {'frame_interval_sec': 2.0},
                {'confidence'        : 0.60},
                {'model_path'        : '/root/yolo26m.pt'},
                {'output_dir'        : OUTPUT_DIR},
                {'min_angle_deg'     : 25.0},
                {'dbscan_eps'        : 0.25},
                {'dbscan_min_samples': 50},
                {'ray_length'        : 8.0},
                {'env_frame_interval': 10},
            ]
        ),

    ])
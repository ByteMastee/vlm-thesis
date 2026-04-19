import os
from launch import LaunchDescription
from launch_ros.actions import Node

# --- Run name: change ONLY here for each new run ---
RUN_NAME = 'run_07'

BASE_OUTPUT_DIR = '/root/UVC_ws/vf_robot_model_ros2/Final_Output/Testing'
OUTPUT_DIR      = os.path.join(BASE_OUTPUT_DIR, RUN_NAME)


def generate_launch_description():
    return LaunchDescription([

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
                {'env_sample_count' : 6},
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
                {'image_topic'       : '/fisheye_front/fisheye_front/image_raw'},
                {'cam_info_topic'    : '/fisheye_front/fisheye_front/camera_info'},
                {'odom_topic'        : '/odom'},
                {'frame_skip'        : 12},
                {'confidence'        : 0.50},
                {'model_path'        : '/root/yolo26m.pt'},
                {'output_dir'        : OUTPUT_DIR},
                {'min_angle_deg'     : 8.0},
                {'dbscan_eps'        : 1.0},
                {'dbscan_min_samples': 3},
                {'ray_length'        : 8.0},
                {'process_delay'     : 2.0},
                {'env_frame_interval': 8},
                {'crop_margin_ratio' : 0.15},   # 0.0 = disabled, 0.15 = drop 15% from each side
                {'ground_truth'      : [
                    'chair_1:-3.0:2.0',
                    'chair_2:-3.5:-2.5',
                    'couch:3.5:0.0',
                    'table:2.0:2.5'
                ]},
            ]
        ),

    ])
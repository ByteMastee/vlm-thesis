import os
from launch import LaunchDescription
from launch_ros.actions import Node

RUN_NAME = 'DC_ViT_Run_01'

BASE_OUTPUT_DIR = '/root/UVC_ws/vf_robot_model_ros2/Final_Output/DualCam'
OUTPUT_DIR      = os.path.join(BASE_OUTPUT_DIR, RUN_NAME)


def generate_launch_description():
    return LaunchDescription([

        Node(
            package    = 'multi_cam_perception',
            executable = 'vlm_label_node_vit',
            name       = 'vlm_label_node_vit',
            output     = 'screen',
            parameters = [
                {'run_name'         : RUN_NAME},
                {'output_dir'       : OUTPUT_DIR},
                {'model_path'       : '/root/UVC_ws/models/qwen2.5-vl-3b'},
                {'max_new_tokens'   : 128},
                {'env_sample_count' : 6},
            ]
        ),

        Node(
            package    = 'multi_cam_perception',
            executable = 'ros_node_vit',
            name       = 'ros_node_vit',
            output     = 'screen',
            parameters = [
                {'run_name'              : RUN_NAME},
                {'image_topic'           : '/fisheye_front/fisheye_front/image_raw'},
                {'cam_info_topic'        : '/fisheye_front/fisheye_front/camera_info'},
                {'image_topic_left'      : '/fisheye_left/fisheye_left/image_raw'},
                {'cam_info_topic_left'   : '/fisheye_left/fisheye_left/camera_info'},
                {'odom_topic'            : '/odom'},
                {'frame_skip'            : 7},
                {'output_dir'            : OUTPUT_DIR},
                {'min_angle_deg'         : 15.0},
                {'dbscan_eps'            : 0.7},
                {'dbscan_min_samples'    : 2},
                {'ray_length'            : 8.0},
                {'process_delay'         : 110.0},
                {'env_frame_interval'    : 8},
                {'min_candidates'        : 3},
                {'sam2_checkpoint'       : '/root/sam2_checkpoints/sam2.1_hiera_small.pt'},
                {'sam2_model_cfg'        : 'configs/sam2.1/sam2.1_hiera_s.yaml'},
                {'points_per_side'       : 16},
                {'pred_iou_thresh'       : 0.82},
                {'stability_score_thresh': 0.85},
                {'min_mask_region_area'  : 1500},
                {'max_mask_area_fraction': 0.20},
                {'max_regions'           : 12},
                {'ground_truth'          : [
                    'chair_1:-3.0:2.0',
                    'chair_2:-3.5:-2.5',
                    'table:2.0:2.5',
                    'couch:3.5:0'
                ]},
            ]
        ),

    ])
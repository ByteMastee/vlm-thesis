from setuptools import find_packages, setup

package_name = 'semantic_mapping'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/semantic_mapping.launch.py']),
        ('share/' + package_name + '/launch', ['launch/vit_vlm_mapping.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='todo@todo.com',
    description='Semantic mapping using YOLO and ROS2',
    license='TODO',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            'ros_node = semantic_mapping.ros_node:main',
            'yolo_map_node = semantic_mapping.yolo_map_node:main',
            'rviz_publisher_node = semantic_mapping.rviz_publisher_node:main',
            'multi_run_visualizer = semantic_mapping.multi_run_visualizer:main',
            'da3_map_node = semantic_mapping.da3_map_node:main',
            'vlm_test_node = semantic_mapping.vlm_test_node:main',
            'qwen_vlm_test_node = semantic_mapping.qwen_vlm_test_node:main',
            'vlm_label_node = semantic_mapping.vlm_label_node:main',
            'vlm_rviz_node = semantic_mapping.vlm_rviz_node:main',
            'ros_node_vit = semantic_mapping.ros_node_vit:main',
            'vlm_label_node_vit = semantic_mapping.vlm_label_node_vit:main',
            'sam2_map_node = semantic_mapping.sam2_map_node:main',
        ],
    },
)

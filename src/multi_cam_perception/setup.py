from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'multi_cam_perception'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='Multi-camera perception package for dual fisheye camera improvement',
    license='TODO: License declaration',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            'ros_node           = multi_cam_perception.ros_node:main',
            'ros_node_vit       = multi_cam_perception.ros_node_vit:main',
            'vlm_label_node     = multi_cam_perception.vlm_label_node:main',
            'vlm_label_node_vit = multi_cam_perception.vlm_label_node_vit:main',
            'vlm_rviz_node      = multi_cam_perception.vlm_rviz_node:main',
        ],
    },
)
from setuptools import setup, find_packages
import os
from glob import glob

package_name = "vf_camera_tf_publisher"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        # Package resource and package.xml
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        # Launch files
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        # Config YAMLs
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        # Service definitions (source .srv files)
        ("share/" + package_name + "/srv", glob("srv/*.srv")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="pravin",
    maintainer_email="olipravin18@gmail.com",
    description="Camera TF publisher and calibration",
    license="Apache License 2.0",
    extras_require={
        "test": ["pytest"],
    },
    entry_points={
        "console_scripts": [
            "publish_camera_tfs = vf_camera_tf_publisher.publish_camera_tfs:main",
            "publish_cam_info = vf_camera_tf_publisher.publish_cam_info:main",
            "calibrate_cams = vf_camera_tf_publisher.calibrate_cams:main",
            "calibrate_intrinsics = vf_camera_tf_publisher.calibrate_intrinsics:main",
            "dummy_camera = vf_camera_tf_publisher.dummy_camera:main",
        ],
    },
)


"""
from setuptools import find_packages, setup

package_name = 'vf_camera_tf_publisher'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='pravin',
    maintainer_email='olipravin18@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
        ],
    },
)
"""

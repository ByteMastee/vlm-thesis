from setuptools import setup
from glob import glob
import os

package_name = "uvc1_gazebo"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        # ament index resource
        (
            "share/ament_index/resource_index/packages",
            [f"resource/{package_name}"],
        ),
        # launch files
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
        # models
        (f"share/{package_name}/models", glob("models/*")),
        # resource files
        (f"share/{package_name}/resource", glob("resource/*")),
        # rviz configs
        (f"share/{package_name}/rviz", glob("rviz/*.rviz")),
        # urdf files
        (f"share/{package_name}/urdf", glob("urdf/*.urdf")),
        # worlds
        (f"share/{package_name}/worlds", glob("worlds/*.world")),
        # package.xml
        (f"share/{package_name}", ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Pravin",
    maintainer_email="olipravin18@gmail.com",
    description="Gazebo simulation package for VF Robot with C++ nodes, Python nodes and plugins.",
    license="Apache-2.0",
    extras_require={
        "test": ["pytest"],
    },
    entry_points={
        "console_scripts": [
            "teleop_twist_keyboard = uvc1_gazebo.scripts.teleop_twist_keyboard:main",
            "ultrasound_py = uvc1_gazebo.scripts.ultrasound_py:main",
        ],
    },
)

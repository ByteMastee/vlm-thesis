from setuptools import setup
from glob import glob

package_name = "vf_robot_model"

setup(
    name=package_name,
    version="0.0.0",
    # explicitly list your Python package
    packages=["vf_robot_model"],
    data_files=[
        # Resource marker so ROS2 knows this is a package
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        # package.xml
        (f"share/{package_name}", ["package.xml"]),
        # Launch files
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="pravin",
    maintainer_email="olipravin18@gmail.com",
    description="VF Robot Model package with C++ and Python ROS2 nodes.",
    license="Apache License 2.0",
    extras_require={
        "test": ["pytest"],
    },
)

from setuptools import setup
from glob import glob

# In a HYBRID ament_cmake + Python package, setup.py has a LIMITED role:
#   - Registers the Python module (uvc1_gazebo/) so `import uvc1_gazebo` works
#   - Installs flat data files (config, launch, maps, etc.)
#
# What CMakeLists.txt handles instead:
#   - C++ compilation and executables
#   - Gazebo plugins
#   - Python scripts → ros2 run  (via file(GLOB) macro)
#     Just drop a .py file in uvc1_gazebo/scripts/ — it becomes a ros2 run
#     executable automatically. No entry here needed. Ever.

package_name = "uvc1_gazebo"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        # ament index marker — required for ros2 pkg list / get_package_share_directory
        (
            "share/ament_index/resource_index/packages",
            [f"resource/{package_name}"],
        ),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/config", glob("config/*.yaml")),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
        (f"share/{package_name}/maps", glob("maps/*")),
        (f"share/{package_name}/models", glob("models/*")),
        (f"share/{package_name}/resource", glob("resource/*")),
        (f"share/{package_name}/rviz", glob("rviz/*.rviz")),
        (f"share/{package_name}/urdf", glob("urdf/*.urdf")),
        (f"share/{package_name}/worlds", glob("worlds/*.world")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Pravin",
    maintainer_email="olipravin18@gmail.com",
    description="Gazebo simulation package for VF Robot with C++ nodes, Python nodes and plugins.",
    license="Apache-2.0",
    extras_require={"test": ["pytest"]},
    entry_points={
        # Python scripts are installed by CMakeLists.txt file(GLOB) macro.
        # Any .py dropped in uvc1_gazebo/scripts/ becomes a ros2 run executable.
        # Nothing needs to be added here.
        "console_scripts": [],
    },
)

from setuptools import find_packages, setup

package_name = 'thesis_pp_develop'

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
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'frame_processing_node = thesis_pp_develop.frames_process:main',
            'ray_visualizing_node = thesis_pp_develop.ray_visualizer:main',
            'rviz_map_publish_node = thesis_pp_develop.rviz_map_publisher:main',
        ],
    },
)

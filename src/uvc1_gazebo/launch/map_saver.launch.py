#!/usr/bin/env python3

from pathlib import Path
import subprocess

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration


def save_map(context):

    maps_folder = Path.home() / "vf_robot_model_ros2/src/uvc1_gazebo/maps"
    maps_folder.mkdir(parents=True, exist_ok=True)

    map_name = LaunchConfiguration("map_name").perform(context)

    if not map_name:
        map_name = "my_map"

    target_folder = maps_folder / map_name
    target_folder.mkdir(parents=True, exist_ok=True)

    db_target = target_folder / f"{map_name}.db"
    map_base = target_folder / map_name

    print("\n========== SAVING MAP ==========")
    print(f"[INFO] Map name : {map_name}")
    print(f"[INFO] Folder   : {target_folder}")
    print("================================\n")

    # ------------------------------------------------
    # GET RTABMAP DATABASE PATH
    # ------------------------------------------------

    try:

        result = subprocess.run(
            ["ros2", "param", "get", "/rtabmap", "database_path"],
            capture_output=True,
            text=True,
        )

        output = result.stdout.strip()

        if "String value is:" in output:

            db_source = output.replace("String value is:", "").strip()

            # expand ~
            db_source = str(Path(db_source).expanduser())

            print(f"[INFO] RTAB-Map DB source: {db_source}")

            if Path(db_source).exists():

                subprocess.run(["cp", db_source, str(db_target)])

                print(f"[INFO] Database copied → {db_target}")

            else:
                print("[WARNING] Database file not found on disk.")

        else:
            print("[WARNING] Could not parse RTAB-Map database path.")

    except Exception as e:
        print("[WARNING] Failed retrieving database path:", e)

    # ------------------------------------------------
    # SAVE OCCUPANCY GRID MAP
    # ------------------------------------------------

    print("\n[INFO] Saving occupancy grid map...")

    subprocess.run(
        [
            "ros2",
            "run",
            "nav2_map_server",
            "map_saver_cli",
            "-f",
            str(map_base),
        ]
    )

    print("\n========== DONE ==========")
    print(f"[INFO] Files saved in: {target_folder}\n")


def generate_launch_description():

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "map_name",
                default_value="my_map",
                description="Map name",
            ),
            OpaqueFunction(function=save_map),
        ]
    )


"""
===============================================================
ROS2 MAP SAVER (RTAB-MAP + NAV2)
===============================================================

This launch file saves a complete map after running RTAB-Map SLAM.

It saves three files:
    1. .db   → Full RTAB-Map SLAM database
    2. .pgm  → 2D occupancy map for Nav2
    3. .yaml → Nav2 metadata for the map

Saved structure:

maps/
 └── map_name/
      map_name.db
      map_name.pgm
      map_name.yaml


---------------------------------------------------------------
HOW TO RUN
---------------------------------------------------------------

Default map name (creates "my_map" folder):

    ros2 launch uvc1_gazebo map_saver.launch.py

Result:

    maps/
     └── my_map
          my_map.db
          my_map.pgm
          my_map.yaml


Custom map name:

    ros2 launch uvc1_gazebo map_saver.launch.py map_name:=house1

Result:

    maps/
     └── house1
          house1.db
          house1.pgm
          house1.yaml


---------------------------------------------------------------
REQUIREMENT
---------------------------------------------------------------

RTAB-Map SLAM must be running before saving the map.

Example SLAM pipeline:

    ros2 launch uvc1_gazebo nav2_rtabmap_launch.py


---------------------------------------------------------------
WHERE RTAB-MAP DATABASE IS STORED
---------------------------------------------------------------

RTAB-Map automatically saves the SLAM database while running.

Default location:

    ~/.ros/rtabmap_multi_camera.db

You can verify the database path:

    ros2 param get /rtabmap database_path


Check that the database exists:

    ls ~/.ros/rtabmap_multi_camera.db


---------------------------------------------------------------
VIEWING THE DATABASE (.db)
---------------------------------------------------------------

The RTAB-Map database contains:

    • RGB images
    • depth images
    • keyframes
    • loop closures
    • pose graph
    • 3D map

Open the database viewer:

    rtabmap-databaseViewer ~/.ros/rtabmap_multi_camera.db


Inside the viewer you can inspect:

    Graph view
    Camera images
    Loop closures
    3D map


---------------------------------------------------------------
NAVIGATION PIPELINE
---------------------------------------------------------------

Typical robotics workflow:

    RTAB-Map SLAM
          ↓
    map_saver.launch.py
          ↓
    maps/
       map_name/
          map_name.db
          map_name.pgm
          map_name.yaml
          ↓
    Nav2 localization + navigation


---------------------------------------------------------------
NOTES
---------------------------------------------------------------

.db
    Full SLAM database (for visualization, relocalization)

.pgm
    2D occupancy grid used by Nav2

.yaml
    Metadata describing the occupancy map

===============================================================
"""

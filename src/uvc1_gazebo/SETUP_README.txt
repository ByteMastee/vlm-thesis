# Autonomous Exploration System — Setup & Usage
# Robot: uvc1_virofighter | Cameras: d455 + d435i | Stack: RTAB-Map + Nav2 + explore_lite

## What this does

1. RTAB-Map builds a 2D occupancy map live using your two RGB-D cameras
2. Nav2 navigates the robot, avoiding obstacles with your camera point clouds
3. explore_lite automatically picks frontier goals (boundaries of known/unknown space)
4. You can ALSO give manual Nav2 goals in RViz — they work alongside auto-exploration
5. Map is saved every 30 seconds and a final map is saved when exploration is complete


## File placement

Copy files into your package like this:

    uvc1_gazebo/
    ├── config/
    │   ├── nav2_params.yaml                    ← unchanged (your existing file)
    │   ├── rtabmap_params_exploration.yaml     ← NEW (replaces rtabmap_params_multicamera.yaml)
    │   └── explore_lite_params.yaml            ← NEW
    ├── launch/
    │   └── exploration_launch.py               ← NEW (the one launch file)
    └── uvc1_gazebo/
        └── exploration_monitor.py              ← NEW (Python node)


## Install explore_lite

    sudo apt install ros-humble-explore-lite
    # or for other distros:
    sudo apt install ros-$ROS_DISTRO-explore-lite


## Register exploration_monitor in setup.py

In your package's setup.py, add to entry_points → console_scripts:

    entry_points={
        'console_scripts': [
            # ... your existing entries ...
            'exploration_monitor = uvc1_gazebo.exploration_monitor:main',
        ],
    },

Then rebuild:

    cd ~/your_ws
    colcon build --packages-select uvc1_gazebo
    source install/setup.bash


## Launch order

Step 1 — Start Gazebo + robot (your existing launch):

    ros2 launch uvc1_gazebo uvc1_my_world_nav2_rtabmap_launch.py

Step 2 — Start exploration (this new launch):

    ros2 launch uvc1_gazebo exploration_launch.py

Step 3 — Watch in RViz:
    - The /map topic fills in as the robot explores
    - You can give Nav2 goals (2D Goal Pose button) at any time on explored areas
    - explore_lite will resume auto-exploration when your goal is reached


## Giving manual goals during exploration

In RViz:
1. Click "2D Goal Pose"
2. Click anywhere on the ALREADY EXPLORED (white/gray) part of the map
3. The robot drives there while still avoiding obstacles
4. When it arrives, explore_lite picks up again automatically

⚠️  Do NOT give goals in the black (unknown) area — Nav2 cannot plan there.


## Stopping exploration manually

    ros2 topic pub /exploration_done std_msgs/msg/Bool "{data: true}" --once

This immediately saves the final map and stops the monitor.


## Map output

Maps are saved to ~/explored_maps/ as:
    auto_20240315_143022.pgm   ← periodic auto-save
    auto_20240315_143022.yaml
    final_20240315_144500.pgm  ← saved when exploration completes
    final_20240315_144500.yaml

The final .yaml + .pgm can be used directly with your existing
localization_launch_multicamera.py:

    ros2 launch uvc1_gazebo localization_launch_multicamera.py map_name:=<your_map_folder>


## Timing of node startup (built into launch file)

    t=0s   → RTAB-Map + rgbd_sync nodes start
    t=5s   → Nav2 starts (waits for /map to exist)
    t=12s  → explore_lite starts (waits for Nav2 action servers)
    t=15s  → exploration_monitor starts


## Troubleshooting

Problem: explore_lite not finding frontiers
→ Check: ros2 topic echo /map  (is RTAB-Map publishing?)
→ Check: ros2 topic hz /map   (should update at ~1 Hz while moving)

Problem: Nav2 says "No path found"
→ The goal is in unknown space. Give goals only on explored (white) areas.

Problem: Robot spinning in place
→ explore_lite picked a bad frontier. It will blacklist it after 30s and try another.

Problem: Map not saving
→ Check ~/explored_maps/ exists (created automatically)
→ Run: ros2 run nav2_map_server map_saver_cli -f /tmp/test_map

Problem: explore_lite and manual goal conflict
→ This is normal — explore_lite will re-send a frontier goal after your manual
  goal completes. If you want to take full manual control, stop explore_lite:
  ros2 lifecycle set /explore_lite deactivate

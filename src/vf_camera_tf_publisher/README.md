# vf_camers_tf_publisher
## 1. Overview

This repository contains the `vf_camera_tf_publisher` ROS 2 package.
It publishes static camera TFs and CameraInfo topics for VF camera setups.

---
## 2. Prerequisites
- ROS 2 Humble
- Docker & Docker Compose
- Linux (tested on Ubuntu 22.04)

---
## 3. Docker Setup
### 3.1 Docker Setup:
Follow the Docker build instructions here:
👉 [Docker Build Guide](../../../README.md)

### 3.2 Start containers:
```bash
cd ~/UVCRobotPC_ROS2
docker-compose up -d
```
### 3.3 Enter container (repeat for each new terminal):
```bash
docker exec -it uvcrobotpc bash
cd /home/robi/UVCRobotPC/ros2_ws
source /opt/ros/humble/setup.bash
```
### Verify ROS:
```bash
ros2 topic list
```
### Expected:
```md
/parameter_events
/rosout
```

---
## 4. Build Workspace
```bash
cd /home/robi/UVCRobotPC/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```
---
## 5. Main Launch Files

### 5.1 Publish CameraInfo:
```bash
ros2 launch vf_camera_tf_publisher publish_tfs.launch.py
```
Expected:
```md
.....
Published CameraInfo for 'fisheye_rear'
Published CameraInfo for 'fisheye_left'
```

### 5.2 Publish Static TFs:
```bash
ros2 launch vf_camera_tf_publisher tf.launch.py
```
Expected:
```md
.....
from 'base_link' to 'front_right_usound_link'
```

This is the **main success path**. After this Everything else is optional.

---
## 6. Quick Verification Checks (Optional)
These steps help verify TFs and CameraInfo publishing.
They are not required for normal operation.

### 6.1 Verify Camera TF Publisher
Terminal 1
```bash
docker exec -it uvcrobotpc bash
cd /home/robi/UVCRobotPC/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 run vf_camera_tf_publisher publish_camera_tfs
```
Expected:
```md
Camera TF Publisher initialized
Published 10 static Tfs
```
### 6.2 Visualize TF Tree
Terminal 2
```bash
docker exec -it uvcrobotpc bash
cd /home/robi/UVCRobotPC/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 run tf2_tools view_frames
```
Output:
```md
frames.pdf
```

### 6.3 Verify CameraInfo Publisher
```bash
ros2 run vf_camera_tf_publisher publish_cam_info
```
---
## 7. Camera Calibration (Optional)
### 7.1 Verify Service:
```bash
ros2 service type /start_calibration
```
Expected:
```md
vfmessages/srv/StartCalibrationIntrinsics
```

### 7.2 Verify YAML:
```bash
nano src/vf_camera_tf_publisher/config/camera_calibration_parameters.yaml
```
Ensure:
```md
dummy_camera:
  rostopic: /camera/color/image_raw
```

### 7.3 Run Dummy Camera:
```bash
ros2 run vf_camera_tf_publisher dummy_camera
```

### 7.4 Call Calibration:
```bash
ros2 service call /start_calibration vfmessages/srv/StartCalibrationIntrinsics \
"{camera_id: 'dummy_camera', output_file: '/tmp/test_calibration.png'}"
```

### 7.5 Check Result:
```bash
ls -l /tmp/test_calibration.png
eog /tmp/test_calibration.png
```

### 7.6 Publish More Frames:
Edit dummy_camera.py:
```bash
for _ in range(20):
    pub.publish(msg)
    rclpy.spin_once(node, timeout_sec=0.1)
```

### 7.7 CameraInfo Verification:
```bash
ros2 topic list | grep camera_info
ros2 topic echo /fisheye_front/camera_info_vf1228
```


### 7.8 Reload calibration:
```bash
ros2 topic pub /camera_calibration_complete std_msgs/Bool "{data: true}"
```

---
## 8. Troubleshooting
If TF not working inside Docker then install
```bash
apt-get update
apt-get install -y ros-humble-tf-transformations
```
---
## 9. Maintenance
Clean Rebuild

```bash
rm -rf build install log
colcon build --symlink-install
source install/setup.bash
```
---
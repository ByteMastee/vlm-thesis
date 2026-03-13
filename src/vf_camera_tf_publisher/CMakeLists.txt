cmake_minimum_required(VERSION 3.8)
project(vf_camera_tf_publisher)

# ----------------------
# Find dependencies
# ----------------------
find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# ----------------------
# Generate the service(s)
# ----------------------
rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/StartCalibrationIntrinsics.srv"
)

# Make sure Python can use the generated service
ament_export_dependencies(rosidl_default_runtime)

# ----------------------
# Install Python modules
# ----------------------
ament_python_install_package(${PROJECT_NAME})

# ----------------------
# Install launch files and config
# ----------------------
install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME}
)

# ----------------------
# Generate ament package
# ----------------------
ament_package()
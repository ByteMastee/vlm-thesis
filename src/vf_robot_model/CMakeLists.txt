cmake_minimum_required(VERSION 3.8)
project(vf_robot_model)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(vfmessages REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# ----------------------------
# C++ Node
# ----------------------------
add_executable(ultrasound_cpp src/ultrasound.cpp)

# include headers
target_include_directories(ultrasound_cpp PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(ultrasound_cpp
  rclcpp
  sensor_msgs
  vfmessages
  std_msgs
  geometry_msgs
)

install(TARGETS
  ultrasound_cpp
  DESTINATION lib/${PROJECT_NAME}
)

# ----------------------
# Install Python modules
# ----------------------
ament_python_install_package(${PROJECT_NAME})

# Install all Python executable scripts (without .py extension)
file(GLOB PYTHON_SCRIPTS "vf_robot_model/scripts/*.py")
foreach(SCRIPT ${PYTHON_SCRIPTS})
  get_filename_component(SCRIPT_NAME "${SCRIPT}" NAME_WE)  # Get name without extension
  install(PROGRAMS
    "${SCRIPT}"
    DESTINATION lib/${PROJECT_NAME}
    RENAME ${SCRIPT_NAME}
  )
endforeach()

# ----------------------
# Install launch files, models, worlds, and config
# ----------------------
install(DIRECTORY
  launch
  models
  worlds
  rviz
  DESTINATION share/${PROJECT_NAME}
)


# ----------------------
# Generate ament package
# ----------------------
ament_package()

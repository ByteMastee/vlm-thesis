cmake_minimum_required(VERSION 3.8)
project(uvc1_gazebo)

# ------------------------------------------------------------------------------
# C++ Standard
# ------------------------------------------------------------------------------
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ------------------------------------------------------------------------------
# Find Dependencies
# ------------------------------------------------------------------------------
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)

find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2 REQUIRED)

find_package(gazebo REQUIRED)
find_package(gazebo_ros REQUIRED)
find_package(gazebo_ros_pkgs REQUIRED)

find_package(vfmessages REQUIRED)

# ------------------------------------------------------------------------------
# Include Directories
# ------------------------------------------------------------------------------
include_directories(
  include
  ${GAZEBO_INCLUDE_DIRS}
)

link_directories(${GAZEBO_LIBRARY_DIRS})

set(COMMON_DEPENDENCIES
  rclcpp
  std_msgs
  sensor_msgs
  geometry_msgs
  nav_msgs
  tf2
  vfmessages
)

# ==============================================================================
# ============================ C++ EXECUTABLES =================================
# ==============================================================================

# turtlebot3_drive node
add_executable(turtlebot3_drive
  src/turtlebot3_drive.cpp
)

ament_target_dependencies(turtlebot3_drive
  ${COMMON_DEPENDENCIES}
)

# ultrasound C++ node
add_executable(ultrasound_cpp
  src/ultrasound.cpp
)

ament_target_dependencies(ultrasound_cpp
  ${COMMON_DEPENDENCIES}
)

target_include_directories(turtlebot3_drive PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_include_directories(ultrasound_cpp PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

# ==============================================================================
# ============================== GAZEBO PLUGINS ================================
# ==============================================================================

add_library(traffic_light_plugin SHARED
  src/traffic_light_plugin.cpp
)
ament_target_dependencies(traffic_light_plugin ${COMMON_DEPENDENCIES})
target_link_libraries(traffic_light_plugin ${GAZEBO_LIBRARIES})

add_library(traffic_bar_plugin SHARED
  src/traffic_bar_plugin.cpp
)
ament_target_dependencies(traffic_bar_plugin ${COMMON_DEPENDENCIES})
target_link_libraries(traffic_bar_plugin ${GAZEBO_LIBRARIES})

add_library(obstacle1 SHARED
  src/obstacle1.cpp
)
ament_target_dependencies(obstacle1 ${COMMON_DEPENDENCIES})
target_link_libraries(obstacle1 ${GAZEBO_LIBRARIES})

add_library(obstacle2 SHARED
  src/obstacle2.cpp
)
ament_target_dependencies(obstacle2 ${COMMON_DEPENDENCIES})
target_link_libraries(obstacle2 ${GAZEBO_LIBRARIES})

add_library(obstacles SHARED
  src/obstacles.cpp
)
ament_target_dependencies(obstacles ${COMMON_DEPENDENCIES})
target_link_libraries(obstacles ${GAZEBO_LIBRARIES})

# ==============================================================================
# ============================== INSTALL SECTION ================================
# ==============================================================================

# Install C++ executables
install(TARGETS
  turtlebot3_drive
  ultrasound_cpp
  DESTINATION lib/${PROJECT_NAME}
)

# Install Gazebo plugins
install(TARGETS
  traffic_light_plugin
  traffic_bar_plugin
  obstacle1
  obstacle2
  obstacles
  LIBRARY DESTINATION lib
)

# Install headers
install(DIRECTORY include/
  DESTINATION include
)

# Install Python package
ament_python_install_package(${PROJECT_NAME})

# Install Python executable scripts
# file(GLOB PYTHON_SCRIPTS
#   "uvc1_gazebo/scripts/*.py"
# )

# foreach(SCRIPT ${PYTHON_SCRIPTS})
#   get_filename_component(SCRIPT_NAME ${SCRIPT} NAME_WE)
#   install(PROGRAMS
#     ${SCRIPT}
#     DESTINATION lib/${PROJECT_NAME}
#     RENAME ${SCRIPT_NAME}
#   )
# endforeach()

# Install config, launch, models, urdf, worlds, rviz
install(DIRECTORY
  config
  launch
  models
  rviz
  urdf
  worlds
  DESTINATION share/${PROJECT_NAME}
)

# ==============================================================================
# ============================== EXPORTS ========================================
# ==============================================================================

ament_export_include_directories(include)

ament_export_dependencies(
  rclcpp
  rclpy
  std_msgs
  sensor_msgs
  geometry_msgs
  nav_msgs
  tf2
  nav2_msgs 
  gazebo_ros
  gazebo_ros_pkgs
  vfmessages
)

ament_package()
cmake_minimum_required(VERSION 3.8)
project(vf_robot_description)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)

# =============================
# Install URDF, meshes, materials
# =============================
install(DIRECTORY urdf
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY meshes
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY materials
  DESTINATION share/${PROJECT_NAME}
)

# =============================
# ament package
# =============================
ament_package()

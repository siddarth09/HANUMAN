cmake_minimum_required(VERSION 3.8)
project(mars_gazebo)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ── Dependencies ──
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(nav_msgs REQUIRED)

# ── ONNX Runtime ──
find_library(ONNXRUNTIME_LIB onnxruntime
  PATHS /usr/local/lib /usr/lib
)
find_path(ONNXRUNTIME_INCLUDE onnxruntime_cxx_api.h
  PATHS /usr/local/include /usr/include
)

if(NOT ONNXRUNTIME_LIB OR NOT ONNXRUNTIME_INCLUDE)
  message(FATAL_ERROR
    "ONNX Runtime not found! Install with:\n"
    "  wget https://github.com/microsoft/onnxruntime/releases/download/v1.21.1/onnxruntime-linux-x64-gpu-1.21.1.tgz\n"
    "  tar -xzf onnxruntime-linux-x64-gpu-1.21.1.tgz\n"
    "  sudo cp -r onnxruntime-linux-x64-gpu-1.21.1/include/* /usr/local/include/\n"
    "  sudo cp -r onnxruntime-linux-x64-gpu-1.21.1/lib/* /usr/local/lib/\n"
    "  sudo ldconfig"
  )
endif()

message(STATUS "ONNX Runtime lib: ${ONNXRUNTIME_LIB}")
message(STATUS "ONNX Runtime include: ${ONNXRUNTIME_INCLUDE}")

# ── rl_policy_node ──
add_executable(rl_policy_node src/rl_policy.cpp)

target_include_directories(rl_policy_node PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${ONNXRUNTIME_INCLUDE}
)

target_link_libraries(rl_policy_node
  ${ONNXRUNTIME_LIB}
)

ament_target_dependencies(rl_policy_node
  rclcpp
  sensor_msgs
  std_msgs
  geometry_msgs
  ament_index_cpp
  nav_msgs
)

target_compile_features(rl_policy_node PUBLIC cxx_std_17)

# ── Install executable ──
install(TARGETS rl_policy_node
  DESTINATION lib/${PROJECT_NAME}
)

# ── Install package directories ──
install(DIRECTORY launch/ DESTINATION share/${PROJECT_NAME}/launch)
install(DIRECTORY config/ DESTINATION share/${PROJECT_NAME}/config)
install(DIRECTORY unitree_g1/ DESTINATION share/${PROJECT_NAME}/unitree_g1)
install(DIRECTORY worlds/ DESTINATION share/${PROJECT_NAME}/worlds)
install(DIRECTORY model/ DESTINATION share/${PROJECT_NAME}/model)

# Install policy (populated by download_policy.py)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/policy")
  install(DIRECTORY policy/ DESTINATION share/${PROJECT_NAME}/policy)
endif()

# Install Python scripts
install(PROGRAMS
  scripts/policy_downloader.py
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
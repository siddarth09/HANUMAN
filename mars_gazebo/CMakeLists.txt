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
# ── LibTorch (TorchScript inference) ──
# If Torch_DIR isn't set, reuse the pip-installed PyTorch's bundled libtorch
# (its CMake config + libs ship inside the python package).
if(NOT DEFINED Torch_DIR AND NOT DEFINED ENV{Torch_DIR})
  execute_process(
    COMMAND python3 -c "import torch,os;print(os.path.join(os.path.dirname(torch.__file__),'share/cmake/Torch'))"
    OUTPUT_VARIABLE _TORCH_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _TORCH_PY_RC)
  if(_TORCH_PY_RC EQUAL 0 AND EXISTS "${_TORCH_CMAKE_DIR}")
    set(Torch_DIR "${_TORCH_CMAKE_DIR}")
  endif()
endif()

find_package(Torch REQUIRED)
message(STATUS "LibTorch: ${TORCH_INSTALL_PREFIX} (flags: ${TORCH_CXX_FLAGS})")

# ── rl_policy_node ──
add_executable(rl_policy_node src/rl_policy.cpp)

target_include_directories(rl_policy_node PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${TORCH_INCLUDE_DIRS}
)

target_link_libraries(rl_policy_node
  ${TORCH_LIBRARIES}
)

# Apply LibTorch's required ABI flag (-D_GLIBCXX_USE_CXX11_ABI=...).
if(TORCH_CXX_FLAGS)
  separate_arguments(_TORCH_FLAGS UNIX_COMMAND "${TORCH_CXX_FLAGS}")
  target_compile_options(rl_policy_node PRIVATE ${_TORCH_FLAGS})
endif()

# Embed the libtorch lib dir in RPATH so the bundled torch/cuda .so resolve at
# runtime without the user setting LD_LIBRARY_PATH.
set_target_properties(rl_policy_node PROPERTIES
  BUILD_WITH_INSTALL_RPATH FALSE
  INSTALL_RPATH "${TORCH_INSTALL_PREFIX}/lib"
  INSTALL_RPATH_USE_LINK_PATH TRUE)

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
install(DIRECTORY unitree_g1_mjcf/ DESTINATION share/${PROJECT_NAME}/unitree_g1_mjcf)
# Install policy (populated by download_policy.py)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/policy")
  install(DIRECTORY policy/ DESTINATION share/${PROJECT_NAME}/policy)
endif()

# Install Python scripts
install(PROGRAMS
  scripts/policy_downloader.py
  scripts/teleop.py
  scripts/odom_tf.py
  scripts/rl_policy_node.py
  scripts/height_scanner_node.py
  scripts/lidar3d_node.py
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
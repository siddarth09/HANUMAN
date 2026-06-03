cmake_minimum_required(VERSION 3.8)
project(mars_gazebo)

# ── Dependencies ──
find_package(ament_cmake REQUIRED)

# ── Install package directories ──
install(DIRECTORY launch/ DESTINATION share/${PROJECT_NAME}/launch)
install(DIRECTORY config/ DESTINATION share/${PROJECT_NAME}/config)
install(DIRECTORY unitree_g1/ DESTINATION share/${PROJECT_NAME}/unitree_g1)
install(DIRECTORY worlds/ DESTINATION share/${PROJECT_NAME}/worlds)
install(DIRECTORY model/ DESTINATION share/${PROJECT_NAME}/model)
install(DIRECTORY unitree_g1_mjcf/ DESTINATION share/${PROJECT_NAME}/unitree_g1_mjcf)
# Install policy (populated by policy_downloader.py)
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

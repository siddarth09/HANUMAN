cmake_minimum_required(VERSION 3.8)
project(mars_terrain_exporter)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

# Install the Python package
ament_python_install_package(${PROJECT_NAME})

# Install the console-script entry point
install(PROGRAMS
  bin/mars_terrain_exporter
  DESTINATION bin
)

install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()

  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_site_catalog test/test_site_catalog.py)
  ament_add_pytest_test(test_site_config test/test_site_config.py)
  ament_add_pytest_test(test_cli test/test_cli.py)
  ament_add_pytest_test(test_mjcf_writer test/test_mjcf_writer.py)
  ament_add_pytest_test(test_procedural test/test_procedural.py)
  ament_add_pytest_test(test_texture test/test_texture.py)
endif()

ament_package()

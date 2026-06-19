"""HANUMAN orbital-prior DEMO: full chain on a rosbag, visualized in RViz.

  ros2 launch terrain_localization terrain_demo.launch.py bag:=/home/sid/projects25/rosbag2_2026_06_16-17_04_02

Brings up: static map->odom (v1: identity) + leg_odom + EKF + GTSAM slam_node +
terrain_matcher_node + RViz, and plays the bag. Watch GTSAM (red) get pulled onto
ground truth (green) by the terrain fixes (yellow) over the HiRISE DEM cloud.
"""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    tl = get_package_share_directory("terrain_localization")
    se = get_package_share_directory("state_estimation")
    bag = LaunchConfiguration("bag")
    rviz = LaunchConfiguration("rviz")

    return LaunchDescription([
        DeclareLaunchArgument("bag", default_value="/home/sid/projects25/rosbag2_2026_06_16-17_04_02"),
        DeclareLaunchArgument("rviz", default_value="true"),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(se, "launch", "state_estimation.launch.py")),
            launch_arguments={"use_sim_time": "true", "run_slam": "true"}.items()),
        # terrain_localization now provides the map->odom TF and (optionally) RViz
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(tl, "launch", "terrain_localization.launch.py")),
            launch_arguments={"use_sim_time": "true", "rviz": rviz}.items()),

        ExecuteProcess(cmd=["ros2", "bag", "play", bag, "--clock"], output="screen"),
    ])

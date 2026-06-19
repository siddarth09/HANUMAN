"""HANUMAN HiRISE orbital prior — terrain-relative localization node.

Brings up terrain_matcher_node: /d435/depth + /leg_odometry + /imu -> /terrain_match/pose
(consumed by state_estimation/slam_node as a GTSAM PriorFactor). Also provides the
map->odom TF (this package defines the global 'map' frame; v1 assumes map==odom at
spawn) and, optionally, RViz.

  ros2 launch terrain_localization terrain_localization.launch.py            # node + TF
  ros2 launch terrain_localization terrain_localization.launch.py rviz:=true # + RViz
"""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory("terrain_localization")
    default_cfg = os.path.join(pkg, "config", "terrain_localization.yaml")
    rviz_cfg = os.path.join(pkg, "rviz", "terrain.rviz")

    use_sim_time = LaunchConfiguration("use_sim_time")
    config_file = LaunchConfiguration("config_file")
    publish_map_tf = LaunchConfiguration("publish_map_tf")
    rviz = LaunchConfiguration("rviz")

    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("config_file", default_value=default_cfg),
        DeclareLaunchArgument("publish_map_tf", default_value="true",
                              description="publish identity map->odom (v1: map==odom at spawn)"),
        DeclareLaunchArgument("rviz", default_value="false",
                              description="also open RViz with the terrain config"),

        Node(
            package="terrain_localization",
            executable="terrain_matcher_node",
            name="terrain_matcher_node",
            output="screen",
            parameters=[config_file, {"use_sim_time": use_sim_time}],
        ),
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="map_to_odom",
            arguments=["0", "0", "0", "0", "0", "0", "map", "odom"],
            condition=IfCondition(publish_map_tf),
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            arguments=["-d", rviz_cfg],
            condition=IfCondition(rviz),
        ),
    ])

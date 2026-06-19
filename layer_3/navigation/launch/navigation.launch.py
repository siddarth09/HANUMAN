"""HANUMAN navigation: cost map + A* planner + MPPI, optional dashboard/RViz."""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    rviz_arg = DeclareLaunchArgument("rviz", default_value="false")
    # shared RViz config from terrain_localization (layer_2)
    rviz_cfg = os.path.join(
        get_package_share_directory("terrain_localization"), "rviz", "terrain.rviz")

    mppi_arg = DeclareLaunchArgument(
        "mppi", default_value="true",
        description="also start the MPPI local planner (-> /cmd_vel)")

    planner = Node(
        package="navigation",
        executable="global_planner_node",
        name="global_planner",
        output="screen",
        parameters=[{"use_sim_time": True}],
    )

    mppi = Node(
        package="navigation",
        executable="mppi_node",
        name="mppi_local_planner",
        output="screen",
        parameters=[{"use_sim_time": True}],
        condition=IfCondition(LaunchConfiguration("mppi")),
    )

    dashboard_arg = DeclareLaunchArgument(
        "dashboard", default_value="false",
        description="open the Qt Mars-ops console")
    dashboard = Node(
        package="navigation",
        executable="dashboard_qt",
        name="hanuman_dashboard",
        output="screen",
        parameters=[{"use_sim_time": True}],
        condition=IfCondition(LaunchConfiguration("dashboard")),
    )

    rviz = Node(
        package="rviz2", executable="rviz2", name="rviz2",
        arguments=["-d", rviz_cfg],
        condition=IfCondition(LaunchConfiguration("rviz")),
    )

    return LaunchDescription(
        [rviz_arg, mppi_arg, dashboard_arg, planner, mppi, dashboard, rviz])

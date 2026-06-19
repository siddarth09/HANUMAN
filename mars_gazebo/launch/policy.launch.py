"""HANUMAN — terrain height scanner + RL policy.

Run this after the sim (mujoco.launch.py) is up. The height scanner starts first
so /height_scan is already publishing before the policy begins inference.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory("mars_gazebo")
    scene_file = os.path.join(pkg_share, "unitree_g1_mjcf", "mars_nav_scene.xml")

    device_arg = DeclareLaunchArgument(
        "device", default_value="cuda", description="Torch device: cuda or cpu")
    model_arg = DeclareLaunchArgument(
        "model_path",
        default_value="/home/sid/projects25/src/HANUMAN/mars_gazebo/policy/model_220000.pt",
        description="Path to the policy .pt")

    height_scanner = Node(
        package="mars_gazebo",
        executable="height_scanner_node.py",
        parameters=[{"use_sim_time": True, "scene_path": scene_file}],
        output="screen",
    )

    rl_policy = Node(
        package="mars_gazebo",
        executable="rl_policy_node.py",
        parameters=[{
            "use_sim_time": True,
            "device": LaunchConfiguration("device"),
            "model_path": LaunchConfiguration("model_path"),
        }],
        output="screen",
    )

    # Height scanner first; bring up the policy 3 s later, once /height_scan is live.
    delayed_policy = TimerAction(period=3.0, actions=[rl_policy])

    return LaunchDescription([device_arg, model_arg, delayed_policy])

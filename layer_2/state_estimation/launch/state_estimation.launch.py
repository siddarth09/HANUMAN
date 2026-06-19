"""
HANUMAN: state estimation launch file.

Brings up the leg-odometry node (joint states + IMU -> body velocity) and the
Error-State EKF node (IMU prediction + leg-odom velocity update -> fused
/odometry/filtered + TF). Both are configured from
state_estimation/config/ekf_config.yaml.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('state_estimation')
    default_config = os.path.join(pkg_share, 'config', 'ekf_config.yaml')

    # The G1 URDF (with floating base) used by leg odometry lives in mars_gazebo.
    default_urdf = os.path.join(
        get_package_share_directory('mars_gazebo'),
        'unitree_g1', 'g1_mujoco.urdf')

    # ---- Launch arguments ----
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time', default_value='true',
        description='Use /clock from the simulator')
    config_arg = DeclareLaunchArgument(
        'config_file', default_value=default_config,
        description='Path to the EKF / leg-odom YAML config')
    urdf_arg = DeclareLaunchArgument(
        'urdf_path', default_value=default_urdf,
        description='Path to the G1 floating-base URDF for leg odometry')

    use_sim_time = LaunchConfiguration('use_sim_time')
    config_file = LaunchConfiguration('config_file')
    urdf_path = LaunchConfiguration('urdf_path')

    # ---- Leg odometry: /joint_states + /imu -> /leg_odometry ----
    leg_odom_node = Node(
        package='state_estimation',
        executable='leg_odom_node',
        name='leg_odom_node',
        parameters=[
            config_file,
            {
                'use_sim_time': use_sim_time,
                'urdf_path': urdf_path,
            },
        ],
        output='screen',
    )

    # ---- Error-State EKF: IMU + /leg_odometry -> /odometry/filtered + TF ----
    ekf_node = Node(
        package='state_estimation',
        executable='ekf_node',
        name='ekf_node',
        parameters=[
            config_file,
            {'use_sim_time': use_sim_time},
        ],
        output='screen',
    )

   
    run_slam = LaunchConfiguration('run_slam')
    run_slam_arg = DeclareLaunchArgument(
        'run_slam', default_value='true',
        description='Bring up the GTSAM factor-graph SLAM node (Layer 2.2)')
    slam_node = Node(
        package='state_estimation',
        executable='slam_node',
        name='slam_node',
        parameters=[
            config_file,
            {'use_sim_time': use_sim_time},
        ],
        condition=IfCondition(run_slam),
        output='screen',
    )

    return LaunchDescription([
        use_sim_time_arg,
        config_arg,
        urdf_arg,
        run_slam_arg,
        leg_odom_node,
        ekf_node,
        slam_node,
    ])

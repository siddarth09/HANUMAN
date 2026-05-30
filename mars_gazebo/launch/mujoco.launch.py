"""
HANUMAN: mujoco_ros2_control launch file for G1 29-DOF.
"""
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_share = get_package_share_directory('mars_gazebo')

    # Paths
    urdf_file = os.path.join(pkg_share, 'unitree_g1', 'g1_mujoco.urdf')
    mjcf_file = os.path.join(pkg_share, 'unitree_g1_mjcf', 'g1_mars.xml')
    controllers_file = os.path.join(pkg_share, 'config', 'controller_mjcf.yaml')

    # Read URDF for robot_state_publisher
    with open(urdf_file, 'r') as f:
        robot_description = f.read()

    # Launch arguments
    headless_arg = DeclareLaunchArgument(
        'headless', default_value='false',
        description='Run MuJoCo without GUI window')
    sim_speed_arg = DeclareLaunchArgument(
        'sim_speed', default_value='1.0',
        description='Simulation speed factor (1.0 = real-time)')

    # robot_state_publisher: publishes /robot_description and /tf from URDF
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{
            'robot_description': robot_description,
            'use_sim_time': True,
        }],
        output='screen',
    )

    imu_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['imu_broadcaster',
                   '--controller-manager', '/controller_manager'],
        output='screen',
    )
    
    delayed_imu = TimerAction(period=7.0, actions=[imu_spawner])

    ft_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['left_foot_ft_broadcaster', 'right_foot_ft_broadcaster',
                   '--controller-manager', '/controller_manager'],
        output='screen',
    )
    delayed_ft = TimerAction(period=7.0, actions=[ft_spawner])

    control_node = Node(
        package='mujoco_ros2_control',
        executable='ros2_control_node',
        parameters=[
            {'use_sim_time': True},
            controllers_file,
        ],
        output='screen',
    )

    # Spawn controllers (with delay to let controller manager start)
    spawn_jsb = TimerAction(
        period=2.0,
        actions=[
            ExecuteProcess(
                cmd=['ros2', 'control', 'load_controller', '--set-state', 'active',
                     'joint_state_broadcaster'],
                output='screen',
            )
        ],
    )

    spawn_pos = TimerAction(
        period=3.0,
        actions=[
            ExecuteProcess(
                cmd=['ros2', 'control', 'load_controller', '--set-state', 'active',
                     'g1_position_controller'],
                output='screen',
            )
        ],
    )


    return LaunchDescription([
        headless_arg,
        sim_speed_arg,
        robot_state_publisher,
        control_node,
        spawn_jsb,
        spawn_pos,
        delayed_imu,
        delayed_ft,
    ])
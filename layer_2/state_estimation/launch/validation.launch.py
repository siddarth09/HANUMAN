"""HANUMAN: state-estimation validation recorder.

Records /ground_truth/odom, /odometry/filtered, /leg_odometry and /joint_states
to CSVs for offline comparison. Fuses nothing and publishes nothing.

  ros2 launch state_estimation validation.launch.py
  # drive the robot, then Ctrl-C -> CSVs + plots in output_dir

Then (re)analyze any recording offline, tuning the contact threshold freely:
  ros2 run state_estimation plot_validation <output_dir> --threshold 8
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')
    output_dir = LaunchConfiguration('output_dir')
    threshold = LaunchConfiguration('contact_effort_threshold')

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true',
                              description='Use /clock from the simulator'),
        DeclareLaunchArgument('output_dir', default_value='output/',
                              description='Where to write CSVs'),
        DeclareLaunchArgument('contact_effort_threshold', default_value='5.0',
                              description='Ankle-effort contact threshold used to shade contact bands'),
        Node(
            package='state_estimation',
            executable='validation_node',
            name='validation_recorder',
            output='screen',
            parameters=[{
                'use_sim_time': use_sim_time,
                'output_dir': output_dir,
                'contact_effort_threshold': threshold,
            }],
        ),
    ])

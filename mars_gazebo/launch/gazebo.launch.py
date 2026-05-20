"""
HANUMAN — Launch G1 in Mars Gazebo Environment.

Launches:
  1. Gazebo Sim with Mars terrain world
  2. robot_state_publisher (URDF → TF tree)
  3. Spawn G1 model into Gazebo
  4. ros2_control controller_manager (via gz_ros2_control plugin)
  5. Joint state broadcaster
  6. Position controller
  7. ros_gz_bridge for sensor topics

Usage:
  ros2 launch mars_gazebo gazebo.launch.py
  ros2 launch mars_gazebo gazebo.launch.py terrain:=flat
  ros2 launch mars_gazebo gazebo.launch.py terrain:=curiosity
  ros2 launch mars_gazebo gazebo.launch.py terrain:=mars_base
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def launch_setup(context):
    pkg_share = get_package_share_directory("mars_gazebo")
    models_path = os.path.join(pkg_share, "model")

    # Set GZ_SIM_RESOURCE_PATH so Gazebo finds our models
    if "GZ_SIM_RESOURCE_PATH" in os.environ:
        os.environ["GZ_SIM_RESOURCE_PATH"] += os.pathsep + models_path
    else:
        os.environ["GZ_SIM_RESOURCE_PATH"] = models_path

    # ── Resolve terrain ──
    terrain_val = context.launch_configurations.get("terrain", "flat")
    world_files = {
        "flat": os.path.join(pkg_share, "worlds", "mars_flat.sdf"),
        "curiosity": os.path.join(pkg_share, "worlds", "mars_curiosity.sdf"),
        "mars_base": os.path.join(pkg_share, "worlds", "mars_base.sdf"),
    }
    world_file = world_files.get(terrain_val, world_files["flat"])

    # ── Terrain-dependent spawn positions ──
    spawn_positions = {
        "flat":      ("0.0", "0.0", "1.0"),
        "curiosity": ("0.0", "0.0", "1.0"),
        "mars_base": ("0.0", "0.0", "5.0"),
    }
    sx, sy, sz = spawn_positions.get(terrain_val, ("0.0", "0.0", "1.0"))

    # ── URDF ──
    urdf_file = os.path.join(pkg_share, "unitree_g1", "g1_gazebo.urdf")
    controllers_yaml = os.path.join(pkg_share, "config", "controller.yaml")
    mesh_dir = os.path.join(pkg_share, "unitree_g1", "meshes")
    with open(urdf_file, "r") as f:
        robot_description = f.read().replace(
            "__CONTROLLERS_YAML_PATH__", controllers_yaml
        ).replace(
            "package://mars_gazebo/unitree_g1/meshes",
            "file://" + mesh_dir
        )

    # ── 1. Gazebo Sim ──
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("ros_gz_sim"),
                "launch",
                "gz_sim.launch.py",
            ])
        ),
        launch_arguments={
            "gz_args": f"-r -v 4 {world_file}",
            "on_exit_shutdown": "true",
        }.items(),
    )

    # ── 2. Robot State Publisher ──
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[{
            "robot_description": robot_description,
            "use_sim_time": True,
        }],
        output="screen",
    )

    # ── 3. Spawn G1 into Gazebo ──
    spawn_robot = Node(
        package="ros_gz_sim",
        executable="create",
        arguments=[
            "-name", "g1",
            "-topic", "robot_description",
            "-x", sx,
            "-y", sy,
            "-z", sz,
        ],
        output="screen",
    )

    # ── 4. ROS-Gazebo Bridge ──
    bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            "/imu/data@sensor_msgs/msg/Imu@gz.msgs.IMU",
            "/camera/depth/image_raw@sensor_msgs/msg/Image@gz.msgs.Image",
            "/camera/depth/camera_info@sensor_msgs/msg/CameraInfo@gz.msgs.CameraInfo",
            "/camera/depth/points@sensor_msgs/msg/PointCloud2@gz.msgs.PointCloudPacked",
            "/camera/color/image_raw@sensor_msgs/msg/Image@gz.msgs.Image",
            "/lidar/points@sensor_msgs/msg/PointCloud2@gz.msgs.PointCloudPacked",
            "/world/mars_surface/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
            "/ground_truth/odom@nav_msgs/msg/Odometry@gz.msgs.Odometry",
        ],
        remappings=[
            ("/world/mars_surface/clock", "/clock"),
        ],
        output="screen",
        parameters=[{"use_sim_time": True}],
    )

    # ── 5. Controller Spawners (delayed for gz_ros2_control startup) ──
    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "joint_state_broadcaster",
            "--controller-manager", "/controller_manager",
        ],
        parameters=[{"use_sim_time": True}],
        output="screen",
    )

    position_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "g1_position_controller",
            "--controller-manager", "/controller_manager",
        ],
        parameters=[{"use_sim_time": True}],
        output="screen",
    )

    return [
        gazebo,
        robot_state_publisher,
        spawn_robot,
        bridge,
        TimerAction(period=5.0, actions=[joint_state_broadcaster_spawner]),
        TimerAction(period=7.0, actions=[position_controller_spawner]),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "terrain",
            default_value="flat",
            description="Terrain: flat, curiosity, mars_base",
            choices=["flat", "curiosity", "mars_base"],
        ),
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="true",
        ),
        OpaqueFunction(function=launch_setup),
    ])
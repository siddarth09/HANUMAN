import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
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
        "empty":            os.path.join(pkg_share, "worlds", "mars_empty.sdf"),
        "flat":             os.path.join(pkg_share, "worlds", "mars.sdf"),
        "curiosity":        os.path.join(pkg_share, "worlds", "mars_curiosity.sdf"),
        "mars_base":        os.path.join(pkg_share, "worlds", "mars_base.sdf")
    }

    # World name INSIDE the SDF — must match for service/topic names
    world_names = {
        "empty":            "mars_surface",
        "flat":             "mars_surface",
        "curiosity":        "mars_surface",
        "mars_base":        "mars_surface"
    }

    spawn_positions = {
        "empty":              ("0.0", "0.0", "0.78"),
        "flat":               ("0.0", "0.0", "0.78"),
        "curiosity":          ("0.0", "0.0", "0.0"),
        "mars_base":          ("0.0", "-3.0", "1.5")
    }

    world_file = world_files.get(terrain_val, world_files["flat"])
    world_name = world_names.get(terrain_val, "mars_surface")
    sx, sy, sz = spawn_positions.get(terrain_val, ("0.0", "0.0", "0.0"))

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
            "gz_args": f"-v 4 {world_file}",
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

    # ── Unpause (uses world_name for correct service path) ──
    unpause_sim = ExecuteProcess(
        cmd=[
            "gz", "service",
            "-s", f"/world/{world_name}/control",
            "--reqtype", "gz.msgs.WorldControl",
            "--reptype", "gz.msgs.Boolean",
            "--timeout", "5000",
            "--req", "pause: false",
        ],
        output="screen",
    )

    # ── Odom → TF broadcaster ──
    odom_to_tf = Node(
        package="mars_gazebo",
        executable="odom_tf.py",
        parameters=[{"use_sim_time": True}],
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

    # ── 4. ROS-Gazebo Bridge (inline, with world_name for clock) ──
    bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            # Clock (world-name dependent)
            f"/world/{world_name}/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
            # IMU
            "/imu/data@sensor_msgs/msg/Imu[gz.msgs.IMU",
            # Odometry
            "/model/g1/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry",
            # Color camera
            "/camera/color@sensor_msgs/msg/Image[gz.msgs.Image",
            # Camera info
            "/camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo",
            # LiDAR
            "/lidar/points/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked",
            # Height scanner
            "/height_scan/points/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked",
        ],
        remappings=[
            (f"/world/{world_name}/clock", "/clock"),
            ("/imu/data", "/imu_broadcaster/imu"),
            ("/model/g1/odometry", "/ground_truth/odom"),
            ("/camera/color", "/camera/color/image_raw"),
            ("/camera/camera_info", "/camera/depth/camera_info"),
            ("/lidar/points/points", "/lidar/points"),
            ("/height_scan/points/points", "/height_scan/points"),
        ],
        output="screen",
        parameters=[{"use_sim_time": True}],
    )

    # ── 5. Controller Spawners ──
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

    # ── 6. Default standing pose ──
    default_pose = [
        # Left leg (6)
        -0.312,  0.0,  0.0,  0.669,  -0.363,  0.0,
        # Right leg (6)
        -0.312,  0.0,  0.0,  0.669,  -0.363,  0.0,
        # Waist (3)
        0.0,  0.0,  0.0,
        # Left arm + elbow (4)
        0.2,  0.2,  0.0,  0.6,
        # Left wrist (3)
        0.0,  0.0,  0.0,
        # Right arm + elbow (4)
        0.2, -0.2,  0.0,  0.6,
        # Right wrist (3)
        0.0,  0.0,  0.0,
    ]
    pose_str = ", ".join(str(v) for v in default_pose)

    def make_pose_cmd():
        return ExecuteProcess(
            cmd=[
                "ros2", "topic", "pub", "--once",
                "/g1_position_controller/commands",
                "std_msgs/msg/Float64MultiArray",
                f"{{data: [{pose_str}]}}",
            ],
            output="screen",
        )

    # ── Launch sequence ──
    # Unpause FIRST so clock flows to controller_manager
    return [
        gazebo,
        robot_state_publisher,
        spawn_robot,
        bridge,
        odom_to_tf,
        TimerAction(period=2.0,  actions=[unpause_sim]),
        TimerAction(period=3.0,  actions=[joint_state_broadcaster_spawner]),
        TimerAction(period=4.0,  actions=[position_controller_spawner]),
        TimerAction(period=5.0,  actions=[make_pose_cmd()]),
        # Repeat at t=8s and t=12s — ensure robot reaches default pose
        # even at low RTF before the policy's 8s warmup guard triggers.
        TimerAction(period=8.0,  actions=[make_pose_cmd()]),
        TimerAction(period=12.0, actions=[make_pose_cmd()]),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "terrain",
            default_value="flat",
            description="Terrain: empty, flat, curiosity, mars_base",
            choices=["empty", "flat", "curiosity", "mars_base"],
        ),
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="true",
        ),
        OpaqueFunction(function=launch_setup),
    ])
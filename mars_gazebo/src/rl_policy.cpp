/**
 * HANUMAN — RL Policy Deployment Node (Implementation)
 *
 * All arrays are in MJLAB TRAINING ORDER. The controller.yaml must also
 * list joints in this same order so commands.data[i] maps 1:1.
 * Joint states arrive in URDF order and are remapped via joint_index_map_.
 */

#include "mars_gazebo/rl_policy.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>

#include <ament_index_cpp/get_package_share_directory.hpp>

namespace fs = std::filesystem;

namespace hanuman
{

// ═══════════════════════════════════════════════════════════════════
// Joint configuration — ALL IN MJLAB TRAINING ORDER
// ═══════════════════════════════════════════════════════════════════

const std::array<std::string, NUM_JOINTS> JOINT_NAMES = {
    "left_hip_pitch_joint",       // 0
    "left_hip_roll_joint",        // 1
    "left_hip_yaw_joint",         // 2
    "left_knee_joint",            // 3
    "left_ankle_pitch_joint",     // 4
    "left_ankle_roll_joint",      // 5
    "right_hip_pitch_joint",      // 6
    "right_hip_roll_joint",       // 7
    "right_hip_yaw_joint",        // 8
    "right_knee_joint",           // 9
    "right_ankle_pitch_joint",    // 10
    "right_ankle_roll_joint",     // 11
    "waist_yaw_joint",            // 12
    "waist_roll_joint",           // 13
    "waist_pitch_joint",          // 14
    "left_shoulder_pitch_joint",  // 15
    "left_shoulder_roll_joint",   // 16
    "left_shoulder_yaw_joint",    // 17
    "left_elbow_joint",           // 18
    "left_wrist_roll_joint",      // 19 
    "left_wrist_pitch_joint",     // 20
    "left_wrist_yaw_joint",       // 21
    "right_shoulder_pitch_joint", // 22
    "right_shoulder_roll_joint",  // 23
    "right_shoulder_yaw_joint",   // 24
    "right_elbow_joint",          // 25
    "right_wrist_roll_joint",     // 26
    "right_wrist_pitch_joint",    // 27
    "right_wrist_yaw_joint",      // 28
};

const std::array<float, NUM_JOINTS> DEFAULT_JOINT_POS = {
    -0.312f,   // 0  left_hip_pitch_joint
     0.0f,     // 1  left_hip_roll_joint
     0.0f,     // 2  left_hip_yaw_joint
     0.669f,   // 3  left_knee_joint
    -0.363f,   // 4  left_ankle_pitch_joint
     0.0f,     // 5  left_ankle_roll_joint
    -0.312f,   // 6  right_hip_pitch_joint
     0.0f,     // 7  right_hip_roll_joint
     0.0f,     // 8  right_hip_yaw_joint
     0.669f,   // 9  right_knee_joint
    -0.363f,   // 10 right_ankle_pitch_joint
     0.0f,     // 11 right_ankle_roll_joint
     0.0f,     // 12 waist_yaw_joint
     0.0f,     // 13 waist_roll_joint
     0.0f,     // 14 waist_pitch_joint
     0.2f,     // 15 left_shoulder_pitch_joint
     0.2f,     // 16 left_shoulder_roll_joint
     0.0f,     // 17 left_shoulder_yaw_joint
     0.6f,     // 18 left_elbow_joint
     0.0f,     // 19 left_wrist_roll_joint
     0.0f,     // 20 left_wrist_pitch_joint
     0.0f,     // 21 left_wrist_yaw_joint
     0.2f,     // 22 right_shoulder_pitch_joint
    -0.2f,     // 23 right_shoulder_roll_joint
     0.0f,     // 24 right_shoulder_yaw_joint
     0.6f,     // 25 right_elbow_joint
     0.0f,     // 26 right_wrist_roll_joint
     0.0f,     // 27 right_wrist_pitch_joint
     0.0f,     // 28 right_wrist_yaw_joint
};

const std::array<float, NUM_JOINTS> ACTION_SCALE = {
    0.5475464629911068f,   // 0  left_hip_pitch_joint
    0.35066146637882434f,  // 1  left_hip_roll_joint
    0.5475464629911068f,   // 2  left_hip_yaw_joint
    0.35066146637882434f,  // 3  left_knee_joint
    0.43857731392336724f,  // 4  left_ankle_pitch_joint
    0.43857731392336724f,  // 5  left_ankle_roll_joint
    0.5475464629911068f,   // 6  right_hip_pitch_joint
    0.35066146637882434f,  // 7  right_hip_roll_joint
    0.5475464629911068f,   // 8  right_hip_yaw_joint
    0.35066146637882434f,  // 9  right_knee_joint
    0.43857731392336724f,  // 10 right_ankle_pitch_joint
    0.43857731392336724f,  // 11 right_ankle_roll_joint
    0.5475464629911068f,   // 12 waist_yaw_joint
    0.43857731392336724f,  // 13 waist_roll_joint
    0.43857731392336724f,  // 14 waist_pitch_joint
    0.43857731392336724f,  // 15 left_shoulder_pitch_joint
    0.43857731392336724f,  // 16 left_shoulder_roll_joint
    0.43857731392336724f,  // 17 left_shoulder_yaw_joint
    0.43857731392336724f,  // 18 left_elbow_joint
    0.43857731392336724f,  // 19 left_wrist_roll_joint
    0.07450087032950714f,  // 20 left_wrist_pitch_joint
    0.07450087032950714f,  // 21 left_wrist_yaw_joint
    0.43857731392336724f,  // 22 right_shoulder_pitch_joint
    0.43857731392336724f,  // 23 right_shoulder_roll_joint
    0.43857731392336724f,  // 24 right_shoulder_yaw_joint
    0.43857731392336724f,  // 25 right_elbow_joint
    0.43857731392336724f,  // 26 right_wrist_roll_joint
    0.07450087032950714f,  // 27 right_wrist_pitch_joint
    0.07450087032950714f,  // 28 right_wrist_yaw_joint
};

// ═══════════════════════════════════════════════════════════════════
// Constructor
// ═══════════════════════════════════════════════════════════════════

RLPolicyNode::RLPolicyNode() : Node("rl_policy_node")
{
    // ── Parameters ──
    this->declare_parameter<std::string>("onnx_path", "");
    this->declare_parameter<double>("policy_rate", 50.0);
    this->declare_parameter<double>("cmd_vel_x", 0.5);
    this->declare_parameter<double>("cmd_vel_y", 0.0);
    this->declare_parameter<double>("cmd_yaw_rate", 0.0);
    this->declare_parameter<double>("action_clip", 1.0);

    // ── Resolve ONNX path ──
    std::string onnx_path = this->get_parameter("onnx_path").as_string();

    if (onnx_path.empty() || !fs::exists(onnx_path)) {
        try {
            std::string pkg_share =
                ament_index_cpp::get_package_share_directory("mars_gazebo");
            onnx_path = pkg_share + "/policy/hanuman_policy.onnx";
        } catch (...) {
            onnx_path = "policy/hanuman_policy.onnx";
        }
    }

    if (!fs::exists(onnx_path)) {
        RCLCPP_FATAL(this->get_logger(),
            "ONNX policy not found at: %s\n"
            "Run download_policy.py first:\n"
            "  python3 download_policy.py --output-dir %s",
            onnx_path.c_str(),
            fs::path(onnx_path).parent_path().c_str());
        throw std::runtime_error("Policy not found");
    }

    // ── Load ONNX model ──
    RCLCPP_INFO(this->get_logger(), "Loading ONNX policy: %s", onnx_path.c_str());

    env_ = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "hanuman_policy");
    Ort::SessionOptions opts;
    opts.SetInterOpNumThreads(1);
    opts.SetIntraOpNumThreads(2);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // CPU only — MLP is small, GPU adds no benefit
    RCLCPP_INFO(this->get_logger(), "Using CPU execution provider");

    session_ = std::make_unique<Ort::Session>(env_, onnx_path.c_str(), opts);
    RCLCPP_INFO(this->get_logger(), "ONNX model loaded successfully");

    // ── Pre-allocate ONNX tensors ──
    memory_info_ = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    obs_shape_ = {1, OBS_DIM};
    obs_tensor_ = Ort::Value::CreateTensor<float>(
        memory_info_, obs_.data(), OBS_DIM, obs_shape_.data(), obs_shape_.size());

    input_names_  = {"observations"};
    output_names_ = {"actions"};

    // ── Initialize state ──
    obs_.fill(0.0f);
    joint_positions_.fill(0.0f);
    joint_velocities_.fill(0.0f);
    base_lin_vel_.fill(0.0f);
    base_ang_vel_.fill(0.0f);
    projected_gravity_ = {0.0f, 0.0f, -1.0f};
    last_action_.fill(0.0f);
    command_.fill(0.0f);

    // ── Subscribers ──
    rclcpp::QoS qos(1);
    qos.best_effort();

    sub_joint_states_ = this->create_subscription<sensor_msgs::msg::JointState>(
        "/joint_states", qos,
        std::bind(&RLPolicyNode::joint_states_cb, this, std::placeholders::_1));

    sub_imu_ = this->create_subscription<sensor_msgs::msg::Imu>(
        "/imu/data", qos,
        std::bind(&RLPolicyNode::imu_cb, this, std::placeholders::_1));

    sub_cmd_vel_ = this->create_subscription<geometry_msgs::msg::Twist>(
        "/cmd_vel", 10,
        std::bind(&RLPolicyNode::cmd_vel_cb, this, std::placeholders::_1));

    sub_odom_ = this->create_subscription<nav_msgs::msg::Odometry>(
    "/odom", qos,
    std::bind(&RLPolicyNode::odom_cb, this, std::placeholders::_1));

    // ── Publisher ──
    pub_commands_ = this->create_publisher<std_msgs::msg::Float64MultiArray>(
        "/g1_position_controller/commands", 10);

    // ── Timer ──
    double rate = this->get_parameter("policy_rate").as_double();
    auto period = std::chrono::duration<double>(1.0 / rate);
    timer_ = this->create_wall_timer(
        std::chrono::duration_cast<std::chrono::nanoseconds>(period),
        std::bind(&RLPolicyNode::policy_step, this));

    RCLCPP_INFO(this->get_logger(),
        "Policy node ready — %.0fHz, %d joints, obs=%d, act=%d",
        rate, NUM_JOINTS, OBS_DIM, ACT_DIM);
}

// ═══════════════════════════════════════════════════════════════════
// Callbacks
// ═══════════════════════════════════════════════════════════════════

void RLPolicyNode::joint_states_cb(
    const sensor_msgs::msg::JointState::SharedPtr msg)
{
    // /joint_states arrives in URDF order. We use a name-based map
    // to store values in our mjlab-ordered arrays.
    if (joint_index_map_.empty()) {
        for (size_t i = 0; i < msg->name.size(); ++i) {
            for (int j = 0; j < NUM_JOINTS; ++j) {
                if (msg->name[i] == JOINT_NAMES[j]) {
                    joint_index_map_[JOINT_NAMES[j]] = static_cast<int>(i);
                    break;
                }
            }
        }
        RCLCPP_INFO(this->get_logger(),
            "Joint state mapping: %zu/%d joints found",
            joint_index_map_.size(), NUM_JOINTS);
    }

    for (int j = 0; j < NUM_JOINTS; ++j) {
        auto it = joint_index_map_.find(JOINT_NAMES[j]);
        if (it != joint_index_map_.end()) {
            int idx = it->second;
            if (idx < static_cast<int>(msg->position.size())) {
                joint_positions_[j] = static_cast<float>(msg->position[idx]);
            }
            if (idx < static_cast<int>(msg->velocity.size())) {
                joint_velocities_[j] = static_cast<float>(msg->velocity[idx]);
            }
        }
    }
    joint_states_received_ = true;
}

void RLPolicyNode::imu_cb(const sensor_msgs::msg::Imu::SharedPtr msg)
{
    base_ang_vel_[0] = static_cast<float>(msg->angular_velocity.x);
    base_ang_vel_[1] = static_cast<float>(msg->angular_velocity.y);
    base_ang_vel_[2] = static_cast<float>(msg->angular_velocity.z);

    // IMU gives acceleration, not velocity. 
    // For a standing robot, acceleration ≈ 0 in body frame (gravity removed by IMU)
    // Use as rough proxy — policy was trained with actual velocity
    base_lin_vel_[0] = static_cast<float>(msg->linear_acceleration.x);
    base_lin_vel_[1] = static_cast<float>(msg->linear_acceleration.y);
    base_lin_vel_[2] = static_cast<float>(msg->linear_acceleration.z);

    // Projected gravity from orientation
    float qw = static_cast<float>(msg->orientation.w);
    float qx = static_cast<float>(msg->orientation.x);
    float qy = static_cast<float>(msg->orientation.y);
    float qz = static_cast<float>(msg->orientation.z);

    projected_gravity_[0] = -2.0f * (qx * qz - qw * qy);
    projected_gravity_[1] = -2.0f * (qy * qz + qw * qx);
    projected_gravity_[2] = -(1.0f - 2.0f * (qx * qx + qy * qy));
    imu_received_ = true;
}

void RLPolicyNode::cmd_vel_cb(const geometry_msgs::msg::Twist::SharedPtr msg)
{
    command_[0] = static_cast<float>(msg->linear.x);
    command_[1] = static_cast<float>(msg->linear.y);
    command_[2] = static_cast<float>(msg->angular.z);
}

void RLPolicyNode::odom_cb(const nav_msgs::msg::Odometry::SharedPtr msg)
{
    // Body-frame linear velocity from odometry
    base_lin_vel_[0] = static_cast<float>(msg->twist.twist.linear.x);
    base_lin_vel_[1] = static_cast<float>(msg->twist.twist.linear.y);
    base_lin_vel_[2] = static_cast<float>(msg->twist.twist.linear.z);
}
// ═══════════════════════════════════════════════════════════════════
// Policy inference
// ═══════════════════════════════════════════════════════════════════

void RLPolicyNode::build_observation()
{
    obs_.fill(0.0f);

    // [0:3] base linear velocity
    obs_[0] = 0.0f;
    obs_[1] = 0.0f;
    obs_[2] = 0.0f;

    // [3:6] base angular velocity
    std::copy(base_ang_vel_.begin(), base_ang_vel_.end(), obs_.begin() + 3);

    // [6:9] projected gravity
    std::copy(projected_gravity_.begin(), projected_gravity_.end(), obs_.begin() + 6);

    // [9:38] joint positions relative to default
    for (int i = 0; i < NUM_JOINTS; ++i) {
        obs_[9 + i] = joint_positions_[i] - DEFAULT_JOINT_POS[i];
    }

    // [38:67] joint velocities
    std::copy(joint_velocities_.begin(), joint_velocities_.end(), obs_.begin() + 38);

    // [67:96] last action
    std::copy(last_action_.begin(), last_action_.end(), obs_.begin() + 67);

    // [96:99] velocity command
    bool has_cmd = (command_[0] != 0.0f || command_[1] != 0.0f || command_[2] != 0.0f);
    if (has_cmd) {
        std::copy(command_.begin(), command_.end(), obs_.begin() + 96);
    } else {
        obs_[96] = static_cast<float>(this->get_parameter("cmd_vel_x").as_double());
        obs_[97] = static_cast<float>(this->get_parameter("cmd_vel_y").as_double());
        obs_[98] = static_cast<float>(this->get_parameter("cmd_yaw_rate").as_double());
    }

    // [99:286] height scan — fill with training mean (flat ground assumption)
    // Zeroing these causes extreme normalized values that break the policy.
    static constexpr float HEIGHT_SCAN_DEFAULT = 0.150467f;
    std::fill(obs_.begin() + 99, obs_.begin() + 286, HEIGHT_SCAN_DEFAULT);

    // [286:288] foot height — fill with training mean
    obs_[286] = 0.047589f;  // left foot
    obs_[287] = 0.047088f;  // right foot
}

void RLPolicyNode::policy_step()
{
    if (!joint_states_received_) {
        return;
    }

    build_observation();

    obs_tensor_ = Ort::Value::CreateTensor<float>(
        memory_info_, obs_.data(), OBS_DIM, obs_shape_.data(), obs_shape_.size());

    auto output = session_->Run(
        Ort::RunOptions{nullptr},
        input_names_.data(), &obs_tensor_, 1,
        output_names_.data(), 1);

    const float* action_data = output[0].GetTensorData<float>();
    double clip_val = this->get_parameter("action_clip").as_double();

    for (int i = 0; i < ACT_DIM; ++i) {
        float a = action_data[i];
        a = std::clamp(a, static_cast<float>(-clip_val), static_cast<float>(clip_val));
        last_action_[i] = a;
    }

    auto cmd_msg = std_msgs::msg::Float64MultiArray();
    cmd_msg.data.resize(NUM_JOINTS);
    for (int i = 0; i < NUM_JOINTS; ++i) {
        cmd_msg.data[i] = static_cast<double>(
            DEFAULT_JOINT_POS[i] + last_action_[i] * ACTION_SCALE[i]);
    }

    pub_commands_->publish(cmd_msg);
}

}  // namespace hanuman

// ═══════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);

    try {
        auto node = std::make_shared<hanuman::RLPolicyNode>();
        rclcpp::spin(node);
    } catch (const std::exception& e) {
        RCLCPP_FATAL(rclcpp::get_logger("rl_policy_node"), "Fatal: %s", e.what());
        return 1;
    }

    rclcpp::shutdown();
    return 0;
}
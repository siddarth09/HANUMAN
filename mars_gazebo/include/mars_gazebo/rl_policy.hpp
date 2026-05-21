/**
 * HANUMAN — RL Policy Deployment Node (Header)
 *
 * Loads the trained ONNX locomotion policy and runs inference at 50Hz,
 * subscribing to joint states and IMU, and publishing joint position
 * commands to the g1_position_controller.
 */

#ifndef HANUMAN__RL_POLICY_NODE_HPP_
#define HANUMAN__RL_POLICY_NODE_HPP_

#include <array>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <onnxruntime_cxx_api.h>

namespace hanuman
{

// ═══════════════════════════════════════════════════════════════════
// Constants
// ═══════════════════════════════════════════════════════════════════

static constexpr int NUM_JOINTS = 29;
static constexpr int OBS_DIM    = 288;
static constexpr int ACT_DIM    = 29;

extern const std::array<std::string, NUM_JOINTS> JOINT_NAMES;
extern const std::array<float, NUM_JOINTS> DEFAULT_JOINT_POS;
extern const std::array<float, NUM_JOINTS> ACTION_SCALE;

// ═══════════════════════════════════════════════════════════════════
// RLPolicyNode
// ═══════════════════════════════════════════════════════════════════

class RLPolicyNode : public rclcpp::Node
{
public:
    explicit RLPolicyNode();

private:
    // ── Callbacks ──
    void joint_states_cb(const sensor_msgs::msg::JointState::SharedPtr msg);
    void imu_cb(const sensor_msgs::msg::Imu::SharedPtr msg);
    void cmd_vel_cb(const geometry_msgs::msg::Twist::SharedPtr msg);
    void odom_cb(const nav_msgs::msg::Odometry::SharedPtr msg);

    // ── Policy ──
    void build_observation();
    void policy_step();

    // ── ONNX Runtime ──
    Ort::Env env_{nullptr};
    std::unique_ptr<Ort::Session> session_;
    Ort::MemoryInfo memory_info_{nullptr};
    Ort::Value obs_tensor_{nullptr};
    std::vector<int64_t> obs_shape_;
    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;

    // ── Observation buffer ──
    std::array<float, OBS_DIM> obs_{};

    // ── Robot state ──
    std::array<float, NUM_JOINTS> joint_positions_{};
    std::array<float, NUM_JOINTS> joint_velocities_{};
    std::array<float, 3> base_lin_vel_{};
    std::array<float, 3> base_ang_vel_{};
    std::array<float, 3> projected_gravity_{};
    std::array<float, NUM_JOINTS> last_action_{};
    std::array<float, 3> command_{};

    bool joint_states_received_ = false;
    bool imu_received_ = false;
    std::unordered_map<std::string, int> joint_index_map_;

    // ── ROS interfaces ──
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr sub_joint_states_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_imu_;
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr sub_cmd_vel_;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr pub_commands_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_odom_;
    rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace hanuman

#endif  // HANUMAN__RL_POLICY_NODE_HPP_
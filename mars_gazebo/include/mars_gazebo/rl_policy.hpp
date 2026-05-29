/**
 * HANUMAN — RL Policy Deployment Node (Header)
 *
 * Loads the trained ONNX locomotion policy and runs inference at 50Hz,
 * subscribing to joint states, IMU, odometry, height scan pointcloud,
 * and publishing joint position commands to the position_controller.
 */

#ifndef HANUMAN__RL_POLICY_NODE_HPP_
#define HANUMAN__RL_POLICY_NODE_HPP_

#include <array>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <nav_msgs/msg/odometry.hpp>

#include <onnxruntime_cxx_api.h>

namespace hanuman
{

static constexpr int NUM_JOINTS = 29;
static constexpr int OBS_DIM    = 288;
static constexpr int ACT_DIM    = 29;

// Height scan grid: 16x10 = 160 terrain + 12 foot = 172 total
// But ONNX expects 187 height + 2 foot = 189 at obs[99:288]
// TODO: verify exact ONNX layout — using 187+2 for now
static constexpr int HEIGHT_SCAN_SIZE = 187;  // obs[99:286]
static constexpr int FOOT_HEIGHT_SIZE = 2;    // obs[286:288]
static constexpr float HEIGHT_SCAN_DEFAULT = 0.150467f;
static constexpr float MAX_RAY_DISTANCE = 5.0f;

extern const std::array<std::string, NUM_JOINTS> JOINT_NAMES;
extern const std::array<float, NUM_JOINTS> DEFAULT_JOINT_POS;
extern const std::array<float, NUM_JOINTS> ACTION_SCALE;

class RLPolicyNode : public rclcpp::Node
{
public:
    explicit RLPolicyNode();

private:
    // ── Callbacks ──
    void joint_states_cb(const sensor_msgs::msg::JointState::SharedPtr msg);
    void imu_cb(const sensor_msgs::msg::Imu::SharedPtr msg);
    void odom_cb(const nav_msgs::msg::Odometry::SharedPtr msg);
    void cmd_vel_cb(const geometry_msgs::msg::Twist::SharedPtr msg);
    void height_scan_cb(const sensor_msgs::msg::PointCloud2::SharedPtr msg);

    // ── Policy ──
    void build_observation();
    void policy_step();

    // ── Height scan processing ──
    void process_height_scan(const sensor_msgs::msg::PointCloud2::SharedPtr msg);

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
    std::array<float, 3> base_lin_vel_world_{};
    std::array<float, 3> base_ang_vel_{};
    std::array<float, 3> projected_gravity_{};
    std::array<float, 4> imu_quat_{};  // [w, x, y, z]
    std::array<float, NUM_JOINTS> last_action_{};
    std::array<float, 3> command_{};

    // ── Height scan state ──
    std::array<float, HEIGHT_SCAN_SIZE> height_scan_{};
    std::array<float, FOOT_HEIGHT_SIZE> foot_heights_{};
    std::mutex height_scan_mutex_;
    bool height_scan_received_ = false;
    float robot_z_ = 0.0f;
    float robot_yaw_ = 0.0f;

    bool joint_states_received_ = false;
    bool imu_received_ = false;
    std::unordered_map<std::string, int> joint_index_map_;

    bool obs_dump_done_ = false;   // one-shot obs diagnostic on first saturation

    // Warmup: don't run inference until robot has settled on the ground
    bool warmup_done_ = false;
    rclcpp::Time warmup_start_{0, 0, RCL_ROS_TIME};
    double warmup_seconds_ = 5.0;
    int last_warmup_log_ = -1;

    // ── ROS interfaces ──
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr sub_joint_states_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_imu_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_odom_;
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr sub_cmd_vel_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_height_scan_;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr pub_commands_;
    rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace hanuman

#endif  // HANUMAN__RL_POLICY_NODE_HPP_
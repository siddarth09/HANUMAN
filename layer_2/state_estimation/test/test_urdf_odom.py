# test_leg_odom.py
# Run on your machine: python3 test_leg_odom.py

import pinocchio as pin
import numpy as np

URDF_PATH = '/home/sid/projects25/src/HANUMAN/mars_gazebo/unitree_g1/g1_mujoco.urdf'  

# ---- Load model ----
model = pin.buildModelFromUrdf(URDF_PATH, pin.JointModelFreeFlyer())
data = model.createData()
print(f"Model: nq={model.nq} nv={model.nv} njoints={model.njoints}")
print(f"q: base_pos(3) + base_quat(4) + joints({model.nq-7})")
print(f"v: base_lin(3) + base_ang(3) + joints({model.nv-6})")

# ---- Print joints ----
print("\n=== Joints ===")
for i in range(model.njoints):
    name = model.names[i]
    qi = model.joints[i].idx_q
    vi = model.joints[i].idx_v
    print(f"  [{i}] {name:40s} q_idx={qi} v_idx={vi}")

# ---- Check foot frames ----
print("\n=== Foot Frames ===")
for fname in ['left_ankle_roll_link', 'right_ankle_roll_link']:
    fid = model.getFrameId(fname)
    found = fid < model.nframes
    print(f"  {fname}: {'id=' + str(fid) if found else 'NOT FOUND'}")

# ---- FK with standing pose ----
js_names = [
    'left_ankle_pitch_joint', 'left_ankle_roll_joint', 'left_elbow_joint',
    'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint',
    'left_knee_joint', 'left_shoulder_pitch_joint', 'left_shoulder_roll_joint',
    'left_shoulder_yaw_joint', 'left_wrist_pitch_joint', 'left_wrist_roll_joint',
    'left_wrist_yaw_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint',
    'right_elbow_joint', 'right_hip_pitch_joint', 'right_hip_roll_joint',
    'right_hip_yaw_joint', 'right_knee_joint', 'right_shoulder_pitch_joint',
    'right_shoulder_roll_joint', 'right_shoulder_yaw_joint',
    'right_wrist_pitch_joint', 'right_wrist_roll_joint', 'right_wrist_yaw_joint',
    'waist_pitch_joint', 'waist_roll_joint', 'waist_yaw_joint'
]

js_pos = [
    -0.3001, -0.0003, 1.2805, -0.2973, 0.0005, -0.0003,
    0.6274, 0.1994, 0.1962, -0.0005, 0.0003, 0.0, -0.0003,
    -0.3001, 0.0003, 1.2805, -0.2972, -0.0005, 0.0003,
    0.6274, 0.1994, -0.1962, 0.0005, 0.0003, 0.0, 0.0003,
    -0.0028, -0.0001, 0.0
]

q = pin.neutral(model)
for name, val in zip(js_names, js_pos):
    jid = model.getJointId(name)
    if jid < model.njoints:
        q[model.joints[jid].idx_q] = val

pin.forwardKinematics(model, data, q)
pin.updateFramePlacements(model, data)

left_id = model.getFrameId('left_ankle_roll_link')
right_id = model.getFrameId('right_ankle_roll_link')

print(f"\n=== FK (base at origin, standing pose) ===")
print(f"Left foot:  {data.oMf[left_id].translation}")
print(f"Right foot: {data.oMf[right_id].translation}")
print(f"  z should be ~-0.45 (foot below pelvis)")

# ---- Test Jacobian ----
J_left = pin.computeFrameJacobian(model, data, q, left_id, pin.LOCAL_WORLD_ALIGNED)
print(f"\n=== Jacobian ===")
print(f"J shape: {J_left.shape}  (should be 6 x {model.nv})")
print(f"J_joints shape: {J_left[:3, 6:].shape}  (linear, joints only)")
print(f"J_joints rank: {np.linalg.matrix_rank(J_left[:3, 6:])}")
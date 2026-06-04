# leg_odometry.py
# Leg odometry: joint states + contact → velocity measurement
# Uses Pinocchio for FK and Jacobian

import numpy as np
import pinocchio as pin
import logging
from typing import Tuple, Optional

logger = logging.getLogger("LegOdom")


class LegOdometry:
    """
    Computes body velocity from foot contact + forward kinematics.
    
    When a foot is on the ground and not slipping:
        v_foot_world = 0
        v_body_world = -R · (J_joints · q̇ + ω × p_foot_body)
    """

    LEFT_LEG_JOINTS = [
        'left_hip_pitch_joint',
        'left_hip_roll_joint',
        'left_hip_yaw_joint',
        'left_knee_joint',
        'left_ankle_pitch_joint',
        'left_ankle_roll_joint',
    ]

    RIGHT_LEG_JOINTS = [
        'right_hip_pitch_joint',
        'right_hip_roll_joint',
        'right_hip_yaw_joint',
        'right_knee_joint',
        'right_ankle_pitch_joint',
        'right_ankle_roll_joint',
    ]

    LEFT_FOOT_FRAME = 'left_ankle_roll_link'
    RIGHT_FOOT_FRAME = 'right_ankle_roll_link'

    def __init__(self, urdf_path: str,
                 force_contact_threshold: float = 30.0):
    
      
        self.model = pin.buildModelFromUrdf(urdf_path, pin.JointModelFreeFlyer())
        self.data = self.model.createData()

        logger.info(f"Pinocchio model: nq={self.model.nq} nv={self.model.nv} "
                    f"njoints={self.model.njoints} nframes={self.model.nframes}")
        logger.info(f"  q layout: [base_pos(3), base_quat(4), joints({self.model.nq - 7})]")
        logger.info(f"  v layout: [base_lin(3), base_ang(3), joints({self.model.nv - 6})]")

        # ---- Get foot frame IDs ----
        self.left_foot_id = self.model.getFrameId(self.LEFT_FOOT_FRAME)
        self.right_foot_id = self.model.getFrameId(self.RIGHT_FOOT_FRAME)
        logger.info(f"Left foot frame: id={self.left_foot_id}")
        logger.info(f"Right foot frame: id={self.right_foot_id}")

        # ---- Build joint name → Pinocchio q/v index mapping ----
        self._joint_pin_map = {} 
        for i in range(1, self.model.njoints):  
            name = self.model.names[i]
            q_idx = self.model.joints[i].idx_q
            v_idx = self.model.joints[i].idx_v
            self._joint_pin_map[name] = (q_idx, v_idx)

       
        self._js_mapping_built = False
        self._left_js_to_pin = []
        self._right_js_to_pin = []

        # ---- Pinocchio config vectors ----
       
        self.q_pin = pin.neutral(self.model)
        self.v_pin = np.zeros(self.model.nv)

       
        self._nv_joints = self.model.nv - 6

        self.force_contact_threshold = force_contact_threshold
        logger.info(f"Contact threshold: {force_contact_threshold} N (foot Fz)")

    def build_joint_mapping(self, js_names: list):
       
        js_name_to_idx = {name: i for i, name in enumerate(js_names)}

        for jname in self.LEFT_LEG_JOINTS:
            if jname in js_name_to_idx and jname in self._joint_pin_map:
                js_idx = js_name_to_idx[jname]
                pq, pv = self._joint_pin_map[jname]
                self._left_js_to_pin.append((js_idx, pq, pv))
            else:
                logger.warning(f"Left leg joint '{jname}' mapping failed")

        for jname in self.RIGHT_LEG_JOINTS:
            if jname in js_name_to_idx and jname in self._joint_pin_map:
                js_idx = js_name_to_idx[jname]
                pq, pv = self._joint_pin_map[jname]
                self._right_js_to_pin.append((js_idx, pq, pv))
            else:
                logger.warning(f"Right leg joint '{jname}' mapping failed")

        self._js_mapping_built = True
        logger.info(f"Joint mapping built: L={len(self._left_js_to_pin)} "
                    f"R={len(self._right_js_to_pin)} joints mapped")

    def detect_contact(self, fz_left: float, fz_right: float) -> Tuple[bool, bool]:
        
        left = abs(fz_left) > self.force_contact_threshold
        right = abs(fz_right) > self.force_contact_threshold
        return left, right

    def compute_velocity(self, js_position: np.ndarray,
                         js_velocity: np.ndarray,
                         left_contact: bool,
                         right_contact: bool,
                         R_body_to_world: np.ndarray,
                         omega_body: np.ndarray
                         ) -> Tuple[Optional[np.ndarray], float]:

        if not self._js_mapping_built:
            logger.warning("Joint mapping not built yet, call build_joint_mapping()")
            return None, 0.0

        
        if not left_contact and not right_contact:
            return None, 0.0

        
        for js_idx, pq, pv in self._left_js_to_pin:
            self.q_pin[pq] = js_position[js_idx]
            self.v_pin[pv] = js_velocity[js_idx]

        for js_idx, pq, pv in self._right_js_to_pin:
            self.q_pin[pq] = js_position[js_idx]
            self.v_pin[pv] = js_velocity[js_idx]

        # ---- FK ----
        pin.forwardKinematics(self.model, self.data, self.q_pin)
        pin.updateFramePlacements(self.model, self.data)

        # ---- Compute velocity from each contact foot ----
        velocities = []

        if left_contact:
            v = self._single_foot_velocity(
                self.left_foot_id, R_body_to_world, omega_body)
            velocities.append(v)

        if right_contact:
            v = self._single_foot_velocity(
                self.right_foot_id, R_body_to_world, omega_body)
            velocities.append(v)

        v_world = np.mean(velocities, axis=0)
        confidence = 1.0 if (left_contact and right_contact) else 0.7

        return v_world, confidence

    def _single_foot_velocity(self, frame_id: int,
                              R_body_to_world: np.ndarray,
                              omega_body: np.ndarray) -> np.ndarray:
        
        # Foot position in body frame (base at origin → oMf IS body frame)
        p_foot_body = self.data.oMf[frame_id].translation.copy()

        # Jacobian: 6 × nv, LOCAL_WORLD_ALIGNED frame
        # Since base is identity, world-aligned = body-aligned
        J_full = pin.computeFrameJacobian(
            self.model, self.data, self.q_pin,
            frame_id, pin.LOCAL_WORLD_ALIGNED
        )

        # Extract joint columns only (skip base columns 0:6)
        # Linear part is rows 0:3
        J_joints = J_full[:3, 6:]

        # Foot velocity from joint motion (in body frame)
        v_joints = self.v_pin[6:]
        v_foot_body = J_joints @ v_joints

        # Rotation correction
        omega_cross_p = np.cross(omega_body, p_foot_body)

        # Body velocity in world frame
        v_body_world = -R_body_to_world @ (v_foot_body + omega_cross_p)

        return v_body_world

    def get_measurement_for_ekf(self, js_position, js_velocity,
                                left_contact, right_contact,
                                R_body_to_world, omega_body,
                                base_noise=0.05
                                ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Returns (15-element measurement, noise) for EKF update, or (None, None)."""
        v_world, confidence = self.compute_velocity(
            js_position, js_velocity, left_contact, right_contact,
            R_body_to_world, omega_body)

        if v_world is None:
            return None, None

        measurement = np.zeros(15)
        measurement[6:9] = v_world

        noise = np.array([base_noise, base_noise, base_noise]) / max(confidence, 0.1)

        return measurement, noise
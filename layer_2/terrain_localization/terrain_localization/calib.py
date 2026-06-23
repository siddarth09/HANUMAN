from __future__ import annotations
import numpy as np
import mujoco

def model_pelvis_cam(scene_path, cam_name="d435", base_body="pelvis"):
    """T_base_cam (camera OPTICAL frame expressed in the base body frame).

    MuJoCo camera frame is +x right, +y up, looks down -z; the optical frame
    (x right, y down, +z forward) = cam_mat @ diag(1, -1, -1).
    """
  
    m = mujoco.MjModel.from_xml_path(scene_path)
    d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    R_opt = d.cam_xmat[cid].reshape(3, 3) @ np.diag([1.0, -1.0, -1.0])
    Twc = np.eye(4)
    Twc[:3, :3] = R_opt
    Twc[:3, 3] = d.cam_xpos[cid]
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, base_body)
    Twb = np.eye(4)
    Twb[:3, :3] = d.xmat[bid].reshape(3, 3)
    Twb[:3, 3] = d.xpos[bid]
    return np.linalg.inv(Twb) @ Twc


def model_pelvis_lidar(scene_path, site_name="mid360_lidar", base_body="pelvis"):
    """T_base_lidar (lidar site frame expressed in the base body frame)."""
    m = mujoco.MjModel.from_xml_path(scene_path)
    d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, site_name)
    Tws = np.eye(4)
    Tws[:3, :3] = d.site_xmat[sid].reshape(3, 3)
    Tws[:3, 3] = d.site_xpos[sid]
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, base_body)
    Twb = np.eye(4)
    Twb[:3, :3] = d.xmat[bid].reshape(3, 3)
    Twb[:3, 3] = d.xpos[bid]
    return np.linalg.inv(Twb) @ Tws


def quat_R(x, y, z, w):
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
        [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)]])


def yaw_of(x, y, z, w):
    return np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))

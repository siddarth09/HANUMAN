import numpy as np 
import logging 
from scipy.spatial.transform import Rotation 
from utils.quaternion import skew_symmetric

logger = logging.getLogger("ESKF")

class ErrorStateEKF:

    IDX_P = slice(0,3) #Position error
    IDX_V = slice(3,6) #Velocity error 
    IDX_TH = slice(6,9) #Orientation error 
    IDX_BA = slice(9,12) #Accel bias error
    IDX_BG = slice(12,15) #Gyro bias error 
    DIM_ERROR = 15 


    # Measurement config indices (12D boolean mask from YAML)
    # [x, y, z, roll, pitch, yaw, vx, vy, vz, wx, wy, wz]
    #  0  1  2   3     4     5    6   7   8   9  10  11

    MEAS_TO_ERROR = [0,1,2,
                     6,7,8,
                     3,4,5,
                     -1,-1,-1]
    
    def __init__(self,config:dict):
        """INITIALIZE from parsed yaml"""

        g = config.get("gravitational_acceleration",9.81)
        self.gravity = np.array([0.0,0.0,-g])

        # Nominal state 

        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.quaternion = np.array([1.0,0.0,0.0,0.0])
        self.bias_accel = np.zeros(3)
        self.bias_gyro = np.zeros(3)


        init_state = config.get("initial_state",None)

        if init_state is not None: 
            s = np.array(init_state,dtype=float)
            self.position = s[0:3].copy()
            self.velocity = s[6:9].copy() 

            rpy = s[3:6]
            q_scipy = Rotation.from_euler('xyz',rpy).as_quat() 
            self.quaternion = np.array([q_scipy[3],q_scipy[0],
                                        q_scipy[1],q_scipy[2]])
            

        # Initial estimate covariance P 

        init_cov = config.get("initial_estimate_covariance",None)

        if init_cov is not None: 
            self.P = np.diag(np.array(init_cov,dtype=float))
        else:
            self.P = np.diag([1e-2]*3 + [1e-2]*3 + [1e-3]*3 + [1e-2]*3 + [1e-3]*3)

        # Process noise parameters 

        self.sigma_accel = config.get('accel_noise_density',0.5)
        self.sigma_gyro = config.get('gyro_noise_density',0.01)
        self.sigma_bias_accel = config.get('accel_bias_random_walk',0.01)
        self.sigma_bias_gyro = config.get('gyro_bias_random_walk',0.001)

        # Filter settings

        self.freq = config.get('frequency',50.0)
        self.sensor_timeout = config.get('sensor_timeout',0.1)
        self.print_diagnostics = config.get('print_diagnostics',False)

        self.sensors = {}

        self.log = {
            'time': [],
            'position': [],
            'velocity': [],
            'euler_deg': [],
            'bias_accel': [],
            'bias_gyro': [],
            'P_diag': [],
            'innovation': [],
            'mahalanobis': [],
            'step_type': [],
        }

        self.initialized = False 
        self._t = 0.0
        self._predict_count = 0
        self._update_count = 0


        logger.info("ESKF created")
        logger.info(f"  gravity: {self.gravity}")
        logger.info(f"  initial pos: {self.position}")
        logger.info(f"  initial vel: {self.velocity}")
        logger.info(f"  P diag: {np.diag(self.P)}")
        logger.info(f"  sigma_a={self.sigma_accel} sigma_g={self.sigma_gyro}")
        logger.info(f"  sigma_ba={self.sigma_bias_accel} sigma_bg={self.sigma_bias_gyro}")
        logger.info(f"  frequency={self.freq} sensor_timeout={self.sensor_timeout}")


    def register_sensor(self,name:str,config:list,
                        noise:np.ndarray = None,
                        differential: bool = False,
                        relative: bool = False,
                        pose_rejection_threshold: float=np.inf,
                        twist_rejection_threshold: float = np.inf):
        
        config_arr = np.array(config,dtype=bool) 
        assert len(config_arr) == 15 ,f"sensor config must be 15 element got {config_arr}"

        active_meas_indices = [] 
        active_error_indices = []

        for i,active in enumerate(config_arr):
            if active:
                error_idx = self.MEAS_TO_ERROR[i]
                if error_idx == -1:
                    logger.warning(f" {name}: config[{i}] is true but has no "
                                   f"error state mapping (angular vel / accel). "
                                   f"This component drives prediction, not update. Skipping.")
                    
                    continue 
                active_meas_indices.append(i)
                active_error_indices.append(error_idx)

        n_active = len(active_error_indices)

        # Building H for this sensor 

        H = np.zeros((n_active,self.DIM_ERROR))
        for row,col in enumerate(active_error_indices):
            H[row,col] = 1.0

        if noise is None: 
            noise = np.eye(n_active)*0.05 
        else:
            noise = np.array(noise,dtype=float)
            if noise.ndim == 1:
                noise = np.diag(noise) 

        self.sensors[name] = {
            'config': config_arr,
            'active_meas_indices': active_meas_indices,
            'active_error_indices': active_error_indices,
            'H': H,
            'R': noise,
            'differential': differential,
            'relative': relative,
            'pose_rejection_threshold': pose_rejection_threshold,
            'twist_rejection_threshold': twist_rejection_threshold,
            'first_measurement': None,    # stored for relative mode
            'prev_measurement': None,     # stored for differential mode
            'last_time': None,
        }

        logger.info(f"Sensor registered: {name}")
        logger.info(f"  config: {config_arr.astype(int)}")
        logger.info(f"  active meas indices: {active_meas_indices}")
        logger.info(f"  → error state indices: {active_error_indices}")
        logger.info(f"  H shape: {H.shape}")
        logger.info(f"  R diag: {np.diag(noise)}")
        logger.info(f"  differential={differential} relative={relative}")

    
    def predict(self,accel_meas:np.ndarray,gyro_meas: np.ndarray,
                dt:float):
        
        if dt <=0 or dt>1.0:
            logger.warning(f"predict() bad dt={dt:.4f},skipping")
            return 
        
        # Correct the imu reading by subtracting the bias 
        accel_corrected = accel_meas - self.bias_accel 
        gyro_corrected = gyro_meas - self.bias_gyro

        # Current Rotation 
        R = self._get_rotation_matrix()
        # Propogate nominal state 

        accel_world = R @ accel_corrected + self.gravity 
        self.position = self.position + self.velocity*dt +0.5*accel_world*dt**2 

        self.velocity = self.velocity + accel_world * dt 

        angle_delta = gyro_corrected *dt 
        self._apply_small_rotation(angle_delta)

        # Building F error-state jacob 

        F = self._build_F_matrix(R,accel_corrected,gyro_corrected,dt)
        # process noise 
        Q = self._build_Q_matrix(R,dt)

        # Propogate covariance
        
        self.P = F@ self.P @ F.T + Q 

        self.P = 0.5*(self.P+self.P.T)

        self._t += dt
        self._predict_count += 1
        self._log_state('predict')

        if self.print_diagnostics and self._predict_count % 200 == 0:
            logger.debug(f"predict #{self._predict_count} t={self._t:.3f}")
            logger.debug(f"  pos={self.position}")
            logger.debug(f"  vel={self.velocity}")
            logger.debug(f"  P_vel={np.diag(self.P)[3:6]}")
            logger.debug(f"  bias_a={self.bias_accel}")
            logger.debug(f"  bias_g={self.bias_gyro}")





    def update(self,sensor_name:str,measurement:np.ndarray,timestamp: float=None):
        """Measurement update step"""

        if sensor_name not in self.sensors:
            logger.error(f"Unknown sensor:{sensor_name}")
            return 
        
        sensor = self.sensors[sensor_name]

        # Extracting measurement 

        meas_indices = sensor['active_meas_indices']
        z = np.array([measurement[i] for i in meas_indices])
        # Handling relative mode
        if sensor['relative']:
            if sensor['first_measurement'] is None: 
                sensor['first_measurement'] = z.copy() 
                logger.info(f"{sensor_name}:stored first measurement for relative mode")
                return 
            
            z = z - sensor['first_measurement']

        # Handling Differential mode
        if sensor['differential']:
            if sensor['prev_measurement'] is None:
                sensor['prev_measurement'] = z.copy()
                sensor['last_time'] = timestamp
                logger.info(f"{sensor_name}: stored first measurement for differential mode")
                return
            if timestamp is not None and sensor['last_time'] is not None:
                dt_meas = timestamp - sensor['last_time']
                if dt_meas > 0:
                    z = (z - sensor['prev_measurement']) / dt_meas
            sensor['prev_measurement'] = z.copy()
            sensor['last_time'] = timestamp


        # Computing innovation 

        z_predicted = self._extract_predicted_measurement(meas_indices)
        innovation = z - z_predicted 

        for idx_in_z,meas_idx in enumerate(meas_indices):
            if 3 <= meas_idx <=5: 
                innovation[idx_in_z] = self._wrap_angle(innovation[idx_in_z])


        # Mahalanbois distance check 

        H = sensor['H']
        R = sensor['R']
        S = H @ self.P @H.T + R 

        try: 
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:

            logger.warning(f"{sensor_name}:singualr innovation covariance,skipping")

            return
        
        mahalanobis = float(np.sqrt(innovation.T @ S_inv @ innovation))
        has_pose = any(i < 6 for i in meas_indices)     # pos or orient
        has_twist = any(6 <= i < 9 for i in meas_indices)  # velocity
        threshold = np.inf
        if has_pose:
            threshold = min(threshold, sensor['pose_rejection_threshold'])
        if has_twist:
            threshold = min(threshold, sensor['twist_rejection_threshold'])

        if mahalanobis > threshold:
            logger.warning(f"{sensor_name}: REJECTED (mahalanobis={mahalanobis:.2f} "
                          f"> threshold={threshold:.2f})")
            self._log_state(f'rejected:{sensor_name}', {
                'innovation': innovation.copy(),
                'mahalanobis': mahalanobis,
            })
            return
        

        # kalman gain 

        K = self.P @ H.T @ S_inv 

        # Compute error state correction 

        delta_x = K @ innovation 

        # Injecting error into nominal state 
        self.position    += delta_x[self.IDX_P]
        self.velocity    += delta_x[self.IDX_V]
        self._apply_small_rotation(delta_x[self.IDX_TH])
        self.bias_accel  += delta_x[self.IDX_BA]
        self.bias_gyro   += delta_x[self.IDX_BG]

        # Updating Covariance 

        I_KH = np.eye(self.DIM_ERROR) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T

        # Force symmetry
        self.P = 0.5 * (self.P + self.P.T)

        # ---- Bookkeeping ----
        self._update_count += 1
        sensor['last_time'] = timestamp

        self._log_state(f'update:{sensor_name}', {
            'innovation': innovation.copy(),
            'mahalanobis': mahalanobis,
            'kalman_gain_norm': float(np.linalg.norm(K)),
        })
        if self.print_diagnostics:
            logger.debug(f"update #{self._update_count} [{sensor_name}]")
            logger.debug(f"  innovation: {innovation}")
            logger.debug(f"  mahalanobis: {mahalanobis:.3f}")
            logger.debug(f"  K norm: {np.linalg.norm(K):.4f}")
            logger.debug(f"  P_vel after: {np.diag(self.P)[3:6]}")



    def _build_F_matrix(self,R:np.ndarray,accel_corrected:np.ndarray,
                        gyro_corrected:np.ndarray,dt: float)->np.ndarray:
        
        F = np.eye(self.DIM_ERROR)
        I3 = np.eye(3)

        a_skew = skew_symmetric(accel_corrected)  # [a]ₓ
        w_skew = skew_symmetric(gyro_corrected)   # [ω]ₓ
        F[self.IDX_P, self.IDX_V] = I3 * dt
        F[self.IDX_V, self.IDX_TH] = -R @ a_skew * dt   # orientation error → velocity error
        F[self.IDX_V, self.IDX_BA]  = -R * dt             # accel bias error → velocity error

        # Row 3: δθ equation
        # δθ_new = (I - [ω]ₓ·dt)·δθ - I·δb_g·dt
        F[self.IDX_TH, self.IDX_TH] = I3 - w_skew * dt  # orientation propagation
        F[self.IDX_TH, self.IDX_BG] = -I3 * dt           # gyro bias error → orientation error

        return F 
    
    def _build_Q_matrix(self, R: np.ndarray, dt: float) -> np.ndarray:
        Q = np.zeros((self.DIM_ERROR, self.DIM_ERROR))
        Q[self.IDX_P, self.IDX_P] = np.eye(3) * (self.sigma_accel * dt**2 / 2)**2
        accel_cov = np.eye(3) * (self.sigma_accel * dt)**2
        Q[self.IDX_V, self.IDX_V] = R @ accel_cov @ R.T

        # Orientation process noise: gyroscope noise
        Q[self.IDX_TH, self.IDX_TH] = np.eye(3) * (self.sigma_gyro * dt)**2

        # Accel bias random walk
        Q[self.IDX_BA, self.IDX_BA] = np.eye(3) * (self.sigma_bias_accel * dt)**2

        # Gyro bias random walk
        Q[self.IDX_BG, self.IDX_BG] = np.eye(3) * (self.sigma_bias_gyro * dt)**2

        return Q

    def _get_rotation_matrix(self)->np.ndarray:
        return Rotation.from_quat(
            self.quaternion[[1,2,3,0]]
        ).as_matrix()
    

    def _quat_euler_deg(self)->np.ndarray:

        return Rotation.from_quat(
            self.quaternion[[1,2,3,0]]
        ).as_euler('xyz',degrees=True)
    
    def _apply_small_rotation(self, delta_theta: np.ndarray):
        """Apply δθ to quaternion. Error-state injection."""
        dq_scipy = Rotation.from_rotvec(delta_theta).as_quat()  # [x,y,z,w]
        dq = np.array([dq_scipy[3], dq_scipy[0],
                        dq_scipy[1], dq_scipy[2]])  # → [w,x,y,z]
        # q ← q ⊗ δq
        q = self.quaternion
        self.quaternion = np.array([
            q[0]*dq[0] - q[1]*dq[1] - q[2]*dq[2] - q[3]*dq[3],
            q[0]*dq[1] + q[1]*dq[0] + q[2]*dq[3] - q[3]*dq[2],
            q[0]*dq[2] - q[1]*dq[3] + q[2]*dq[0] + q[3]*dq[1],
            q[0]*dq[3] + q[1]*dq[2] - q[2]*dq[1] + q[3]*dq[0],
        ])
        self.quaternion /= np.linalg.norm(self.quaternion)


    def _extract_predicted_measurement(self, meas_indices: list) -> np.ndarray:
        
        euler = Rotation.from_quat(
            self.quaternion[[1, 2, 3, 0]]
        ).as_euler('xyz')

        full_pred = np.zeros(15)
        full_pred[0:3] = self.position
        full_pred[3:6] = euler
        full_pred[6:9] = self.velocity
        # [9:15] angular vel and accel — zero (not in state)

        return np.array([full_pred[i] for i in meas_indices])
    


    def _log_state(self, step_type: str, extra: dict = None):
        """Append current state to log."""
        self.log['time'].append(self._t)
        self.log['position'].append(self.position.copy())
        self.log['velocity'].append(self.velocity.copy())
        self.log['euler_deg'].append(self._quat_euler_deg())
        self.log['bias_accel'].append(self.bias_accel.copy())
        self.log['bias_gyro'].append(self.bias_gyro.copy())
        self.log['P_diag'].append(np.diag(self.P).copy())
        self.log['step_type'].append(step_type)

        if extra:
            for key, val in extra.items():
                if key not in self.log:
                    self.log[key] = []
                self.log[key].append(val)


    def get_state(self) -> dict:
        """Return current estimate. Consumers read this."""
        return {
            'position': self.position.copy(),
            'velocity': self.velocity.copy(),
            'quaternion': self.quaternion.copy(),
            'rotation_matrix': self._get_rotation_matrix(),
            'euler_deg': self._quat_euler_deg(),
            'bias_accel': self.bias_accel.copy(),
            'bias_gyro': self.bias_gyro.copy(),
            'P_diagonal': np.diag(self.P).copy(),
        }
    
    def _wrap_angle(self, angle:float)->float:
        return (angle + np.pi) % (2 * np.pi) - np.pi 
    
    


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    config = {
        'gravitational_acceleration': 9.81,
        'initial_state': [0,0,0.75, 0,0,0, 0,0,0, 0,0,0, 0,0,0],
        'initial_estimate_covariance': [1e-2]*3 + [1e-2]*3 + [1e-3]*3 + [1e-2]*3 + [1e-3]*3,
        'accel_noise_density': 0.5,
        'gyro_noise_density': 0.01,
        'accel_bias_random_walk': 0.01,
        'gyro_bias_random_walk': 0.001,
        'frequency': 50.0,
        'sensor_timeout': 0.1,
        'print_diagnostics': True,
    }
    ekf = ErrorStateEKF(config)

    ekf.register_sensor(
        name='odom0',
        config=[False]*6 + [True,True,True] + [False]*6,
        noise=np.array([0.05, 0.05, 0.05]),
        twist_rejection_threshold=5.0,
    )

    # === Test: Predict-only vs Predict+Update ===
    dt_imu = 0.005        # 200 Hz
    dt_odom = 0.02        # 50 Hz (every 4th IMU step)

    accel = np.array([0.0, 0.0, 9.81])    # standing still
    gyro = np.array([0.0, 0.0, 0.0])

    # Measurement: leg odom says velocity is [0,0,0]
    # (15-element vector, only indices 6,7,8 matter)
    leg_odom_meas = np.zeros(15)

    print("\n=== 2 seconds: predict only (no updates) ===")
    for i in range(600):
        ekf.predict(accel, gyro, dt_imu)
    print(f"  P_vel: {np.diag(ekf.P)[3:6]}")

    print("\n=== Reset and redo with predict + update ===")
    ekf2 = ErrorStateEKF(config)
    ekf2.register_sensor('odom0',
        config=[False]*6 + [True,True,True] + [False]*6,
        noise=np.array([0.05, 0.05, 0.05]))

    for i in range(600):
        ekf2.predict(accel, gyro, dt_imu)
        # Update at 50Hz (every 4th step)
        if i % 4 == 3:
            ekf2.update('odom0', leg_odom_meas)

    print(f"  P_vel: {np.diag(ekf2.P)[3:6]}")
    print(f"\n  Compare: predict-only P_vel grew to {np.diag(ekf.P)[3:6][0]:.4f}")
    print(f"  With updates, P_vel bounded at {np.diag(ekf2.P)[3:6][0]:.4f}")
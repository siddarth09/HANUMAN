from __future__ import annotations
import gtsam
import numpy as np
from gtsam import (
    ISAM2, ISAM2Params, NavState, NonlinearFactorGraph,
    Point3, Pose3, PreintegratedCombinedMeasurements,
    PreintegrationCombinedParams, Rot3, Values,
)
from gtsam.symbol_shorthand import B, V, X

_I3 = np.eye(3)


class ImuLegGraph:
    def __init__(self, *, gravity=9.81, accel_noise=0.5, gyro_noise=0.01,
                 accel_bias_rw=0.01, gyro_bias_rw=0.001, integration_sigma=1e-3,
                 leg_vel_sigma=0.15, init_pos=(0.0, 0.0, 0.75), init_yaw=0.0):

        params = PreintegrationCombinedParams.MakeSharedU(gravity)
        params.setAccelerometerCovariance(_I3 * accel_noise**2)
        params.setGyroscopeCovariance(_I3 * gyro_noise**2)
        params.setIntegrationCovariance(_I3 * integration_sigma**2)
        params.setBiasAccCovariance(_I3 * accel_bias_rw**2)
        params.setBiasOmegaCovariance(_I3 * gyro_bias_rw**2)
        params.setBiasAccOmegaInit(np.eye(6) * 1e-5)

        self._bias = gtsam.imuBias.ConstantBias()
        self.pim = PreintegratedCombinedMeasurements(params, self._bias)
        self.leg_vel_sigma = leg_vel_sigma

        ip = ISAM2Params()
        ip.setRelinearizeThreshold(0.01)
        ip.relinearizeSkip = 1
        self.isam = ISAM2(ip)

        self.k = 0
        self.pose = Pose3(Rot3.Yaw(init_yaw), Point3(*init_pos))
        self.vel = np.zeros(3)

        graph = NonlinearFactorGraph()
        init = Values()
        graph.add(gtsam.PriorFactorPose3(
            X(0), self.pose,
            gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-3] * 6))))
        graph.add(gtsam.PriorFactorVector(
            V(0), self.vel, gtsam.noiseModel.Isotropic.Sigma(3, 1e-3)))
        graph.add(gtsam.PriorFactorConstantBias(
            B(0), self._bias, gtsam.noiseModel.Isotropic.Sigma(6, 1e-3)))
        init.insert(X(0), self.pose)
        init.insert(V(0), self.vel)
        init.insert(B(0), self._bias)
        self.isam.update(graph, init)

    def integrate(self, accel, gyro, dt):
        if dt > 0.0:
            self.pim.integrateMeasurement(np.asarray(accel), np.asarray(gyro), dt)

    def dt_since_keyframe(self):
        return self.pim.deltaTij()

    def add_keyframe(self, leg_vel_world=None):
        i, j = self.k, self.k + 1
        graph = NonlinearFactorGraph()
        init = Values()

        graph.add(gtsam.CombinedImuFactor(X(i), V(i), X(j), V(j), B(i), B(j), self.pim))

        # Initial guess for j = forward-integrate the IMU from i.
        pred = self.pim.predict(NavState(self.pose, self.vel), self._bias)
        init.insert(X(j), pred.pose())
        init.insert(V(j), pred.velocity())
        init.insert(B(j), self._bias)

        # Leg odometry = world-frame body velocity -> prior on V(j).
        if leg_vel_world is not None:
            graph.add(gtsam.PriorFactorVector(
                V(j), np.asarray(leg_vel_world, dtype=float),
                gtsam.noiseModel.Isotropic.Sigma(3, self.leg_vel_sigma)))

        self.isam.update(graph, init)
        est = self.isam.calculateEstimate()
        self.pose = est.atPose3(X(j))
        self.vel = est.atVector(V(j))
        self._bias = est.atConstantBias(B(j))
        self.pim.resetIntegrationAndSetBias(self._bias)   # start a fresh window
        self.k = j
        return self.pose, self.vel

    def add_terrain_prior(self, x, y, yaw, sig_x, sig_y, sig_yaw, z=None, sig_z=1e3):
        # z (terrain-relative DEM height) optional; absent -> current estimate, loose sigma
        if z is None:
            z = self.pose.translation()[2]
        prior = Pose3(Rot3.Yaw(yaw), Point3(x, y, z))
        # Pose3 tangent order is [roll, pitch, yaw, x, y, z]
        sigmas = np.array([1e3, 1e3, sig_yaw, sig_x, sig_y, sig_z])
        graph = NonlinearFactorGraph()
        graph.add(gtsam.PriorFactorPose3(
            X(self.k), prior, gtsam.noiseModel.Diagonal.Sigmas(sigmas)))
        try:
            # empty Values: prior attaches to the existing keyframe X(k), no new variables
            self.isam.update(graph, Values())
            est = self.isam.calculateEstimate()
            self.pose = est.atPose3(X(self.k))
            self.vel = est.atVector(V(self.k))
            return True
        except Exception:
            # e.g. IndeterminantLinearSystem — drop this fix, keep dead-reckoning
            return False

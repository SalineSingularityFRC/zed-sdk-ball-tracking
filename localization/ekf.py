import json
from pathlib import Path

from pupil_apriltags import Detector
import numpy as np

import gtsam
from gtsam import Pose3, Rot3, Point3, Quaternion

from filterpy.kalman import ExtendedKalmanFilter as FpEKF


def load_tag_world_poses(tags_path, allowed_tags=None):
    """Load AprilTag world poses from a tags.json file.

    tags_path: path to a JSON file with a "tags" list of {ID, pose:{translation, rotation:{quaternion}}}.
    allowed_tags: optional iterable of tag IDs to keep; all others are rejected.
    Returns: {tag_id: Pose3} in the world frame.
    """
    with open(Path(tags_path), 'r') as f:
        data = json.load(f)

    allowed = set(allowed_tags) if allowed_tags is not None else None

    tag_world_poses = {}
    for entry in data.get('tags', []):
        tag_id = entry['ID']
        if allowed is not None and tag_id not in allowed:
            continue

        t = entry['pose']['translation']
        q = entry['pose']['rotation']['quaternion']
        rot = Rot3(Quaternion(q['W'], q['X'], q['Y'], q['Z']))
        tag_world_poses[tag_id] = Pose3(rot, Point3(t['x'], t['y'], t['z']))

    return tag_world_poses


class ImuAprilTagEKF:
    """Error-state EKF on SE(3).

    Nominal pose `T_nom` is stored as a GTSAM Pose3. filterpy carries a 6D
    body-frame error state `ξ` (held at zero except transiently during an
    update) and the 6x6 covariance. GTSAM supplies the process model
    (PreintegratedImuMeasurements) and the analytic SE(3) derivatives
    (AdjointMap, Logmap, Expmap, between).

    Error convention: T_true = T_nom ∘ Exp(ξ).
    """

    def __init__(self, init_pose: Pose3, init_cov: np.ndarray,
                 camera_matrix, tag_size,
                 tags_path, allowed_tags=None,
                 accel_noise=0.01, gyro_noise=0.001):

        self.tag_size = tag_size
        self.K = camera_matrix
        self.detector = Detector(families='tag36h11')
        self.tag_world_poses = load_tag_world_poses(tags_path, allowed_tags)

        # IMU preintegration params
        self.imu_params = gtsam.PreintegrationParams.MakeSharedU(9.81)
        self.imu_params.setAccelerometerCovariance(np.eye(3) * accel_noise**2)
        self.imu_params.setGyroscopeCovariance(np.eye(3) * gyro_noise**2)
        self.imu_params.setIntegrationCovariance(np.eye(3) * 1e-8)

        self.bias = gtsam.imuBias.ConstantBias()
        self.velocity = np.zeros(3)
        self.pim = gtsam.PreintegratedImuMeasurements(self.imu_params, self.bias)

        # Nominal pose lives outside the filter
        self.T_nom = init_pose

        # filterpy EKF carries the error state and covariance
        self.ekf = FpEKF(dim_x=6, dim_z=6)
        self.ekf.x = np.zeros(6)
        self.ekf.P = np.asarray(init_cov, dtype=float)
        self.ekf.F = np.eye(6)

        # Process-noise floor (rot, trans), matches previous predict_noise sigmas
        predict_sigmas = np.array([0.001, 0.001, 0.001, 0.01, 0.01, 0.01])
        self.predict_extra_Q = np.diag(predict_sigmas ** 2)

        # AprilTag measurement covariance (rot, trans)
        tag_sigmas = np.array([0.02, 0.02, 0.02, 0.05, 0.05, 0.05])
        self.R_tag = np.diag(tag_sigmas ** 2)

    def integrate_imu(self, accel: np.ndarray, gyro: np.ndarray, dt: float):
        """Call at IMU rate (~200 Hz). Accumulates into PIM."""
        self.pim.integrateMeasurement(accel, gyro, dt)

    def predict(self):
        """Flush PIM into the error-state EKF predict step."""
        nav_state_0 = gtsam.NavState(self.T_nom, self.velocity)
        nav_state_1 = self.pim.predict(nav_state_0, self.bias)

        delta = self.T_nom.between(nav_state_1.pose())
        self.velocity = nav_state_1.velocity()

        # Body-frame error-state process Jacobian: F = Ad(delta^-1)
        F = delta.inverse().AdjointMap()

        # Pose block of the preintegration covariance (NavState ordering: R, p, v)
        Q_pim = np.asarray(self.pim.preintMeasCov())[0:6, 0:6]

        self.T_nom = self.T_nom.compose(delta)
        self.ekf.F = F
        self.ekf.Q = Q_pim + self.predict_extra_Q
        self.ekf.predict()

        self.pim.resetIntegration()

    def update(self, image: np.ndarray):
        """Run AprilTag detections and fold each into the filter."""
        gray = image if image.ndim == 2 else image.mean(axis=2).astype(np.uint8)
        detections = self.detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=[self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]],
            tag_size=self.tag_size,
        )

        for det in detections:
            tag_id = det.tag_id
            if tag_id not in self.tag_world_poses:
                continue

            # Tag pose in camera frame from detector
            R = Rot3(det.pose_R)
            t = Point3(det.pose_t.flatten())
            tag_in_cam = Pose3(R, t)

            tag_in_world = self.tag_world_poses[tag_id]

            # Expected camera pose in world: T_world_cam = T_world_tag * T_tag_cam
            expected_cam_pose = tag_in_world.compose(tag_in_cam.inverse())

            # Innovation lives in the tangent space at the current nominal
            innovation = gtsam.Pose3.Logmap(self.T_nom.between(expected_cam_pose))

            # H = I, predicted measurement = 0 (body-frame error state)
            self.ekf.R = self.R_tag
            self.ekf.update(
                z=innovation,
                HJacobian=lambda x: np.eye(6),
                Hx=lambda x: np.zeros(6),
            )

            # Inject the correction into the nominal and reset the error state
            self.T_nom = self.T_nom.compose(gtsam.Pose3.Expmap(self.ekf.x))
            self.ekf.x = np.zeros(6)

    def pose(self) -> Pose3:
        return self.T_nom

    def covariance(self) -> np.ndarray:
        return self.ekf.P

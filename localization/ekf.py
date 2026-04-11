import json
from pathlib import Path

from pupil_apriltags import Detector
import numpy as np

import gtsam
from gtsam import (
    ExtendedKalmanFilter,
    Pose3, Rot3, Point3, Quaternion,
    noiseModel
)


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

        # GTSAM EKF — operates on Pose3
        # Use symbol 'x0' as the state key
        self.key = gtsam.symbol('x', 0)
        prior_noise = noiseModel.Gaussian.Covariance(init_cov)
        self.ekf = ExtendedKalmanFilter(self.key, init_pose, prior_noise)

        # Noise models
        self.predict_noise = noiseModel.Diagonal.Sigmas(
            np.array([0.001, 0.001, 0.001, 0.01, 0.01, 0.01])  # rot, trans
        )
        self.tag_noise = noiseModel.Diagonal.Sigmas(
            np.array([0.02, 0.02, 0.02, 0.05, 0.05, 0.05])  # rot, trans
        )

    def integrate_imu(self, accel: np.ndarray, gyro: np.ndarray, dt: float):
        """Call at IMU rate (~200 Hz). Accumulates into PIM."""
        self.pim.integrateMeasurement(accel, gyro, dt)

    def predict(self):
        """
        Call at camera rate to flush PIM into EKF predict step.
        Uses a BetweenFactorPose3 as the process model factor.
        """
        # Get relative pose from preintegration
        current_pose = self.ekf.mean()  # Pose3

        nav_state_0 = gtsam.NavState(current_pose, self.velocity)
        nav_state_1 = self.pim.predict(nav_state_0, self.bias)

        delta_pose = current_pose.between(nav_state_1.pose())
        self.velocity = nav_state_1.velocity()

        # Build a BetweenFactorPose3 as the EKF process factor
        process_factor = gtsam.BetweenFactorPose3(
            self.key, self.key,  # EKF handles the key increment internally
            delta_pose,
            self.predict_noise
        )

        self.ekf.predict(process_factor)

        # Reset PIM for next integration window
        self.pim.resetIntegration()

    # ------------------------------------------------------------------
    # AprilTag: update when tags are visible
    # ------------------------------------------------------------------

    def update(self, image: np.ndarray):
        """
        Call at camera rate when a new frame arrives.
        Uses the tag map loaded from tags.json at construction time.
        """
        gray = image if image.ndim == 2 else image.mean(axis=2).astype(np.uint8)
        detections = self.detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=[self.K[0,0], self.K[1,1], self.K[0,2], self.K[1,2]],
            tag_size=self.tag_size
        )

        for det in detections:
            tag_id = det.tag_id
            if tag_id not in self.tag_world_poses:
                continue

            # Pose of tag in camera frame from detector
            R = Rot3(det.pose_R)
            t = Point3(det.pose_t.flatten())
            tag_in_cam = Pose3(R, t)

            # Known tag pose in world
            tag_in_world = self.tag_world_poses[tag_id]

            # Expected camera pose in world: T_world_cam = T_world_tag * T_tag_cam
            expected_cam_pose = tag_in_world.compose(tag_in_cam.inverse())

            # Build a PriorFactorPose3 as the measurement factor
            measurement_factor = gtsam.PriorFactorPose3(
                self.key,
                expected_cam_pose,
                self.tag_noise
            )

            self.ekf.update(measurement_factor)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def pose(self) -> Pose3:
        return self.ekf.mean()

    def covariance(self) -> np.ndarray:
        return self.ekf.covariance()
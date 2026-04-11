import pyzed.sl as sl
import numpy as np
import cv2
from gtsam import Pose3

from .ekf import ImuAprilTagEKF


class CameraLocalizer:
    def __init__(self, zed=None, tag_size=0.1651,
                 tags_path=None, allowed_tags=None,
                 init_pose=None, init_cov=None,
                 accel_noise=0.01, gyro_noise=0.001):
        if zed is None:
            zed = sl.Camera()
            init_params = sl.InitParameters()
            zed.open(init_params)

        self.zed = zed

        # Retrieve camera information
        cam_info = self.zed.get_camera_information()
        calib = cam_info.camera_configuration.calibration_parameters

        self.K_left = np.array([
            [calib.left_cam.fx, 0, calib.left_cam.cx],
            [0, calib.left_cam.fy, calib.left_cam.cy],
            [0, 0, 1]
        ])

        self.K_right = np.array([
            [calib.right_cam.fx, 0, calib.right_cam.cx],
            [0, calib.right_cam.fy, calib.right_cam.cy],
            [0, 0, 1]
        ])

        extrinsics = calib.stereo_transform  # sl.Transform (4x4)
        self.T_left_to_right = extrinsics.m

        sensors_config = cam_info.sensors_configuration
        imu_transform = sensors_config.imu_parameters.pose  # sl.Transform (4x4)
        self.T_left_to_imu = imu_transform.m

        # EKF: defaults to identity pose with moderate prior covariance
        if init_pose is None:
            init_pose = Pose3()
        if init_cov is None:
            sigmas = np.array([0.1, 0.1, 0.1, 0.5, 0.5, 0.5])
            init_cov = np.diag(sigmas ** 2)

        self.ekf = ImuAprilTagEKF(
            init_pose=init_pose,
            init_cov=init_cov,
            camera_matrix=self.K_left,
            tag_size=tag_size,
            tags_path=tags_path,
            allowed_tags=allowed_tags,
            accel_noise=accel_noise,
            gyro_noise=gyro_noise,
        )

        self._image_mat = sl.Mat()
        self._sensors_data = sl.SensorsData()
        self._runtime = sl.RuntimeParameters()
        self._last_imu_ts = None

    def pose(self) -> 'Pose3':
        return self.ekf.pose()

    def covariance(self) -> np.ndarray:
        return self.ekf.covariance()

    def _drain_imu(self):
        """Pull the latest IMU sample from ZED and integrate into the PIM."""
        if self.zed.get_sensors_data(
            self._sensors_data, sl.TIME_REFERENCE.IMAGE
        ) != sl.ERROR_CODE.SUCCESS:
            return

        imu = self._sensors_data.get_imu_data()
        ts = imu.timestamp.get_nanoseconds()
        if self._last_imu_ts is None or ts <= self._last_imu_ts:
            self._last_imu_ts = ts
            return

        dt = (ts - self._last_imu_ts) * 1e-9
        self._last_imu_ts = ts

        lin = imu.get_linear_acceleration()
        ang = imu.get_angular_velocity()
        accel = np.array([lin[0], lin[1], lin[2]], dtype=np.float64)
        # ZED reports angular velocity in deg/s; GTSAM wants rad/s
        gyro = np.deg2rad(np.array([ang[0], ang[1], ang[2]], dtype=np.float64))

        self.ekf.integrate_imu(accel, gyro, dt)

    def step(self):
        """
        Grab a frame from the ZED, integrate IMU since last step,
        run the EKF predict, then update with any visible AprilTags.
        Returns the current Pose3, or None if the grab failed.
        """
        if self.zed.grab(self._runtime) != sl.ERROR_CODE.SUCCESS:
            return None

        self._drain_imu()
        self.ekf.predict()

        self.zed.retrieve_image(self._image_mat, sl.VIEW.LEFT)
        frame = self._image_mat.get_data()
        if frame is not None:
            if frame.ndim == 3 and frame.shape[2] == 4:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
            elif frame.ndim == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
            self.ekf.update(gray)

        return self.ekf.pose()

    def run(self):
        while True:
            pose = self.step()
            if pose is None:
                break


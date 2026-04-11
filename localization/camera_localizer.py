import contextlib
import os
import sys

import pyzed.sl as sl
import numpy as np
import cv2
from pupil_apriltags import Detector
from gtsam import Pose3, Rot3, Point3

from .ekf import AprilTagEKF, load_tag_world_poses, TAG_FRC_TO_DET_POSE


@contextlib.contextmanager
def _silence_stderr():
    """Silence C-level stderr (e.g. apriltag's 'new minima' spam) for the block."""
    sys.stderr.flush()
    saved_fd = os.dup(2)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull_fd, 2)
        yield
    finally:
        os.dup2(saved_fd, 2)
        os.close(devnull_fd)
        os.close(saved_fd)


def _sl_pose_to_pose3(sl_pose: sl.Pose) -> Pose3:
    """Convert a pyzed sl.Pose to a GTSAM Pose3."""
    rot = sl_pose.get_rotation_matrix(sl.Rotation()).r  # 3x3 numpy array
    t = sl_pose.get_translation(sl.Translation()).get()  # [x, y, z]
    return Pose3(Rot3(rot), Point3(t[0], t[1], t[2]))


def _zed_cov_to_gtsam(flat_cov: np.ndarray) -> np.ndarray:
    """Reorder ZED's flat 36-element pose covariance to GTSAM [rot, trans] order.

    ZED covariance order: [tx, ty, tz, rx, ry, rz]
    GTSAM error-state order: [rx, ry, rz, tx, ty, tz]
    """
    C = flat_cov.reshape(6, 6)
    idx = [3, 4, 5, 0, 1, 2]
    return C[np.ix_(idx, idx)]


class CameraLocalizer:
    def __init__(self, zed=None, tag_size=0.1651,
                 tags_path=None, allowed_tags=None,
                 init_pose=None, init_cov=None,
                 predict_sigmas=None):
        if zed is None:
            zed = sl.Camera()
            init_params = sl.InitParameters()
            init_params.coordinate_units = sl.UNIT.METER
            zed.open(init_params)

        self.zed = zed

        # Enable ZED positional tracking (visual-inertial odometry)
        tracking_params = sl.PositionalTrackingParameters()
        self.zed.enable_positional_tracking(tracking_params)

        # Retrieve camera intrinsics
        cam_info = self.zed.get_camera_information()
        calib = cam_info.camera_configuration.calibration_parameters

        self.K_left = np.array([
            [calib.left_cam.fx, 0, calib.left_cam.cx],
            [0, calib.left_cam.fy, calib.left_cam.cy],
            [0, 0, 1]
        ])

        # EKF defaults
        if init_pose is None:
            init_pose = Pose3()
        if init_cov is None:
            sigmas = np.array([0.1, 0.1, 0.1, 0.5, 0.5, 0.5])
            init_cov = np.diag(sigmas ** 2)

        self.ekf = AprilTagEKF(
            init_pose=init_pose,
            init_cov=init_cov,
            camera_matrix=self.K_left,
            tag_size=tag_size,
            tags_path=tags_path,
            allowed_tags=allowed_tags,
            predict_sigmas=predict_sigmas,
        )

        self._image_mat = sl.Mat()
        self._runtime = sl.RuntimeParameters()
        self._zed_pose = sl.Pose()
        self._last_zed_pose3: Pose3 = None

    def pose(self) -> 'Pose3':
        p = self.ekf.pose()
        t = p.translation()
        print(f"[EKF out]    t=({t[0]:+.4f}, {t[1]:+.4f}, {t[2]:+.4f})  "
              f"P_diag={np.diag(self.ekf.covariance()).round(4)}")
        return p

    def covariance(self) -> np.ndarray:
        return self.ekf.covariance()

    def step(self):
        """
        Grab a frame, use ZED's fused pose estimate as the EKF prediction,
        then update with any visible AprilTags.
        Returns the current Pose3, or None if the grab failed.
        """
        if self.zed.grab(self._runtime) != sl.ERROR_CODE.SUCCESS:
            return None

        # --- Prediction from ZED fused visual-inertial odometry ---
        state = self.zed.get_position(self._zed_pose, sl.REFERENCE_FRAME.WORLD)
        if state == sl.POSITIONAL_TRACKING_STATE.OK:
            T_curr = _sl_pose_to_pose3(self._zed_pose)
            if self._last_zed_pose3 is not None:
                delta = self._last_zed_pose3.between(T_curr)
                dt = delta.translation()
                print(f"[ZED delta]  dt=({dt[0]:+.4f}, {dt[1]:+.4f}, {dt[2]:+.4f})  "
                      f"conf={self._zed_pose.pose_confidence:.0f}")
                Q = _zed_cov_to_gtsam(self._zed_pose.pose_covariance)
                self.ekf.predict(delta, Q)
            t = T_curr.translation()
            print(f"[ZED abs]    t=({t[0]:+.4f}, {t[1]:+.4f}, {t[2]:+.4f})")
            self._last_zed_pose3 = T_curr

        # --- Measurement update from AprilTags ---
        self.zed.retrieve_image(self._image_mat, sl.VIEW.LEFT)
        frame = self._image_mat.get_data()
        if frame is not None:
            if frame.ndim == 3 and frame.shape[2] == 4:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
            elif frame.ndim == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
            with _silence_stderr():
                self.ekf.update(gray)

        return self.ekf.pose()

    def step_external(self, frame):
        """Advance the localizer using a frame already grabbed by the caller.

        The caller owns ``self.zed`` and has already called ``grab()`` for the
        current frame, so we skip grabbing/retrieving and just read the fused
        ZED pose and run the AprilTag update on the provided image.
        """
        state = self.zed.get_position(self._zed_pose, sl.REFERENCE_FRAME.WORLD)
        if state == sl.POSITIONAL_TRACKING_STATE.OK:
            T_curr = _sl_pose_to_pose3(self._zed_pose)
            if self._last_zed_pose3 is not None:
                delta = self._last_zed_pose3.between(T_curr)
                Q = _zed_cov_to_gtsam(self._zed_pose.pose_covariance)
                self.ekf.predict(delta, Q)
            self._last_zed_pose3 = T_curr

        if frame is not None:
            if frame.ndim == 3 and frame.shape[2] == 4:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
            elif frame.ndim == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
            with _silence_stderr():
                self.ekf.update(gray)

        return self.ekf.pose()

    def run(self):
        while True:
            pose = self.step()
            if pose is None:
                break


class AprilTagOnlyLocalizer:
    """Camera pose directly from a single AprilTag detection, no EKF fusion."""

    def __init__(self, zed=None, tag_size=0.1651,
                 tags_path=None, allowed_tags=None):
        if zed is None:
            zed = sl.Camera()
            init_params = sl.InitParameters()
            init_params.coordinate_units = sl.UNIT.METER
            zed.open(init_params)

        self.zed = zed

        cam_info = self.zed.get_camera_information()
        calib = cam_info.camera_configuration.calibration_parameters
        self.K_left = np.array([
            [calib.left_cam.fx, 0, calib.left_cam.cx],
            [0, calib.left_cam.fy, calib.left_cam.cy],
            [0, 0, 1]
        ])

        self.tag_size = tag_size
        self.detector = Detector(families='tag36h11')
        self.tag_world_poses = load_tag_world_poses(tags_path, allowed_tags)

        self._image_mat = sl.Mat()
        self._runtime = sl.RuntimeParameters()
        self._last_pose: Pose3 = None

    def step(self):
        """Grab a frame, detect AprilTags, return camera pose from the closest tag.

        Returns the latest pose, or None if the grab failed. Keeps the last
        known pose when no tag is visible.
        """
        if self.zed.grab(self._runtime) != sl.ERROR_CODE.SUCCESS:
            return None

        self.zed.retrieve_image(self._image_mat, sl.VIEW.LEFT)
        frame = self._image_mat.get_data()
        if frame is None:
            return self._last_pose

        if frame.ndim == 3 and frame.shape[2] == 4:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
        elif frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        detections = self.detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=[self.K_left[0, 0], self.K_left[1, 1],
                           self.K_left[0, 2], self.K_left[1, 2]],
            tag_size=self.tag_size,
        )

        best_pose = None
        best_dist = float('inf')
        best_id = None
        for det in detections:
            if det.tag_id not in self.tag_world_poses:
                continue
            dist = float(np.linalg.norm(det.pose_t))
            if dist >= best_dist:
                continue
            # Detector returns the tag expressed in the camera frame (T_cam_tag),
            # and in the detector's own tag-axis convention. Rebase the FRC
            # tag-in-world pose into that same convention before composing.
            tag_in_cam = Pose3(Rot3(det.pose_R), Point3(det.pose_t.flatten()))
            tag_in_world = self.tag_world_poses[det.tag_id].compose(TAG_FRC_TO_DET_POSE)
            best_pose = tag_in_world.compose(tag_in_cam.inverse())
            best_dist = dist
            best_id = det.tag_id

        if best_pose is not None:
            t = best_pose.translation()
            print(f"[tag {best_id:2d}]    cam_t=({t[0]:+.4f}, {t[1]:+.4f}, {t[2]:+.4f})  dist={best_dist:.3f}")
            self._last_pose = best_pose

        return self._last_pose

    def pose(self) -> Pose3:
        return self._last_pose

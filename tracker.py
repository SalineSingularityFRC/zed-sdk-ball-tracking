import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from models import Track
from zed_utils import sl, _has_zed

class WorldCoordinateSystem:
    """Manages the camera-to-world transformation using AprilTags."""
    def __init__(self, tag_size=0.165):
        self.tag_size = tag_size
        self.camera_mtx = None
        self.dist_coeffs = np.zeros((4, 1))
        # Default to no transformation (camera == world)
        self.R = np.eye(3)
        self.t = np.zeros((3, 1))
        
        # AprilTag Setup (using OpenCV ArUco with AprilTag 36h11 dictionary)
        try:
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
            self.aruco_params = cv2.aruco.DetectorParameters()
            if hasattr(cv2.aruco, 'ArucoDetector'):
                self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
            else:
                self.detector = None
        except Exception:
            self.aruco_dict = None
            self.detector = None
            
        self.world_plane_z = 0.5 # 0.5 meters above the tag

    def update_pose(self, frame, zed_intrinsics):
        if zed_intrinsics is None or self.aruco_dict is None:
            return frame
            
        self.camera_mtx = np.array([
            [zed_intrinsics['fx'], 0, zed_intrinsics['cx']],
            [0, zed_intrinsics['fy'], zed_intrinsics['cy']],
            [0, 0, 1]
        ])
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.detector is not None:
            corners, ids, rejected = self.detector.detectMarkers(gray)
        else:
            corners, ids, rejected = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)
            
        vis = frame.copy()
        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(vis, corners, ids)
            # Find the first tag to act as the world origin
            obj_points = np.array([
                [-self.tag_size/2, self.tag_size/2, 0],
                [self.tag_size/2, self.tag_size/2, 0],
                [self.tag_size/2, -self.tag_size/2, 0],
                [-self.tag_size/2, -self.tag_size/2, 0]
            ])
            success, rvec, tvec = cv2.solvePnP(obj_points, corners[0][0], self.camera_mtx, self.dist_coeffs)
            if success:
                cv2.drawFrameAxes(vis, self.camera_mtx, self.dist_coeffs, rvec, tvec, self.tag_size)
                self.R, _ = cv2.Rodrigues(rvec)
                self.t = tvec
                
                # Draw the 0.5m plane visually on frame
                pts_in_world = np.array([
                    [-1.0, 1.0, self.world_plane_z],
                    [1.0, 1.0, self.world_plane_z],
                    [1.0, -1.0, self.world_plane_z],
                    [-1.0, -1.0, self.world_plane_z]
                ])
                pts_in_cam = (self.R @ pts_in_world.T + self.t).T
                img_pts, _ = cv2.projectPoints(pts_in_cam, np.zeros((3,1)), np.zeros((3,1)), self.camera_mtx, self.dist_coeffs)
                img_pts = np.int32(img_pts).reshape(-1, 2)
                cv2.polylines(vis, [img_pts], True, (255, 0, 255), 2)
                cv2.putText(vis, "Target Plane (0.5m)", tuple(img_pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                
        return vis

    def cam_to_world(self, pt_cam):
        if pt_cam is None: return None
        pt_w = self.R.T @ (np.array(pt_cam).reshape(3,1) - self.t)
        return pt_w.flatten().tolist()
        
    def check_plane_intersection(self, pt_world_prev, pt_world_new):
        if pt_world_prev is None or pt_world_new is None:
            return None
        z1 = pt_world_prev[2]
        z2 = pt_world_new[2]
        # Crosses from above plane (z > 0.5) to below plane (z < 0.5)
        # Note: in OpenCV coordinate system, Z might point differently depending on solvePnP, 
        # usually Z is into the tag. If world_plane_z is 0.5, we check crossing that threshold.
        if (z1 >= self.world_plane_z and z2 <= self.world_plane_z) or (z1 <= self.world_plane_z and z2 >= self.world_plane_z):
            # Interpolate
            t = (self.world_plane_z - z1) / (z2 - z1 + 1e-6)
            x_int = pt_world_prev[0] + t * (pt_world_new[0] - pt_world_prev[0])
            y_int = pt_world_prev[1] + t * (pt_world_new[1] - pt_world_prev[1])
            return (x_int, y_int, self.world_plane_z)
        return None

class BallTracker:
    """
    Multi-target ball tracker using greedy trajectory growing with
    Hungarian (optimal) assignment and ballistic motion gating.
    """

    def __init__(self, lower_hsv=None, upper_hsv=None,
                 min_radius=5, max_radius=300,
                 gate_radius=80, max_missed=15,
                 min_track_length=3, min_circularity=0.70,
                 fps=30.0, max_fit_error=15.0, min_displacement=20.0,
                 roi=None):
        self.lower_hsv = np.array(lower_hsv if lower_hsv else [0, 100, 100])
        self.upper_hsv = np.array(upper_hsv if upper_hsv else [20, 255, 255])
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.min_circularity = min_circularity

        self.gate_radius = gate_radius
        self.max_missed = max_missed
        self.min_track_length = min_track_length
        self.fps = fps
        self.max_fit_error = max_fit_error
        self.min_displacement = min_displacement
        self.roi = roi  # (x, y, w, h) or None

        self.tracks: list[Track] = []
        self.frame_count = 0
        self.roi_stats: list[dict] = []  # stats for completed flights through ROI
        self._roi_seen_ids: set[int] = set()  # track IDs already recorded
        # ZED-related state (initialized when tracker is used with a ZED Camera)
        self.zed_cam = None
        self._zed_runtime = None
        self._zed_pc_mat = None
        self.world_system = WorldCoordinateSystem()
        self.intersections = []

    def segment(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_hsv, self.upper_hsv)

        kernel_open = np.ones((5, 5), np.uint8)
        kernel_close = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)
        kernel_erode = np.ones((3, 3), np.uint8)
        mask = cv2.erode(mask, kernel_erode, iterations=1)
        return mask

    def _detect_circles(self, mask):
        """Returns list of (x, y, radius)."""
        circles = []
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < np.pi * (self.min_radius ** 2) * 0.8:
                continue

            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue

            circularity = 4 * np.pi * area / (perimeter ** 2)
            if circularity < self.min_circularity:
                continue

            if len(contour) >= 5:
                (_, (minor_axis, major_axis), _) = cv2.fitEllipse(contour)
                if minor_axis / (major_axis + 1e-6) < 0.65:
                    continue

            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            circle_area = np.pi * radius * radius
            if area / (circle_area + 1e-6) < 0.60:
                continue

            if self.min_radius <= radius <= self.max_radius:
                circles.append((float(cx), float(cy), float(radius)))

        return circles

    # ------------------------------------------------------------------ #
    #  Core tracking: predict → associate → update/create/prune
    # ------------------------------------------------------------------ #

    def track(self, frame, mask):
        self.frame_count += 1
        detections = self._detect_circles(mask)  # list of (x, y, r)

        # --- Predict positions for all active tracks ---
        predictions = []  # parallel to self.tracks
        for tr in self.tracks:
            pred = tr.predict(self.frame_count)
            predictions.append(pred)

        # --- Build cost matrix and solve assignment ---
        n_tracks = len(self.tracks)
        n_dets = len(detections)
        assigned_det = set()
        assigned_track = set()

        if n_tracks > 0 and n_dets > 0:
            cost = np.full((n_tracks, n_dets), 1e9)
            for i, pred in enumerate(predictions):
                if pred is None:
                    continue
                for j, (dx, dy, dr) in enumerate(detections):
                    dist = np.hypot(pred[0] - dx, pred[1] - dy)
                    if dist < self.gate_radius:
                        cost[i, j] = dist

            row_ind, col_ind = linear_sum_assignment(cost)
            for i, j in zip(row_ind, col_ind):
                if cost[i, j] < self.gate_radius:
                    dx, dy, dr = detections[j]
                    # Try to get 3D point if ZED depth is available
                    pos3d = None
                    try:
                        if getattr(self, '_zed_pc_mat', None) is not None:
                            pos3d = self._get_point3d(dx, dy)
                    except Exception:
                        pos3d = None
                    self.tracks[i].update(dx, dy, dr, self.frame_count, pos3d=pos3d)
                    assigned_det.add(j)
                    assigned_track.add(i)

        # --- Mark unassigned tracks as missed ---
        for i, tr in enumerate(self.tracks):
            if i not in assigned_track:
                tr.mark_missed()

        # --- Create new tracks from unassigned detections ---
        for j, (dx, dy, dr) in enumerate(detections):
            if j not in assigned_det:
                # when creating a new track, try to include 3D point
                pos3d = None
                try:
                    if getattr(self, '_zed_pc_mat', None) is not None:
                        pos3d = self._get_point3d(dx, dy)
                except Exception:
                    pos3d = None
                t = Track(dx, dy, dr, self.frame_count)
                # if we have a 3D point, overwrite the last appended None with the real 3D
                if pos3d is not None:
                    # replace last position3d entry in trajectory
                    if t.trajectory.positions3d:
                        t.trajectory.positions3d[-1] = tuple(pos3d)
                self.tracks.append(t)

        # --- Prune dead tracks and spurious trajectories ---
        # Capture ROI stats for tracks about to die (so we get the full trajectory)
        kept = []
        for t in self.tracks:
            if t.missed_frames > self.max_missed or self._is_spurious(t):
                if self.roi is not None:
                    self._capture_roi_stats(t)
            else:
                kept.append(t)
        self.tracks = kept

        # --- Visualize ---
        return self._visualize(frame, detections)

    def _track_hits_roi(self, track):
        """Check if any detection in the track falls inside the ROI."""
        if self.roi is None:
            return False
        rx, ry, rw, rh = self.roi
        for px, py in track.trajectory.positions:
            if rx <= px <= rx + rw and ry <= py <= ry + rh:
                return True
        return False

    def _capture_roi_stats(self, track):
        """Record stats for a valid flight that passed through the ROI."""
        if track.id in self._roi_seen_ids:
            return
        if not self._is_valid_flight(track):
            return
        if not self._track_hits_roi(track):
            return
        self._roi_seen_ids.add(track.id)
        traj = track.trajectory
        airtime = (traj.last_frame - traj.first_frame) / self.fps
        # Prefer computing speed in meters/second if 3D positions are available
        speed = 0.0
        if hasattr(traj, 'positions3d') and traj.positions3d and any(p is not None for p in traj.positions3d[-2:]):
            # find last two valid 3D points
            valid = [p for p in traj.positions3d if p is not None]
            if len(valid) >= 2:
                p0 = np.array(valid[-2])
                p1 = np.array(valid[-1])
                dt_frames = traj.frames[-1] - traj.frames[-2] if len(traj.frames) >= 2 else 1
                dt = dt_frames / self.fps if self.fps > 0 else 1.0
                if dt > 0:
                    speed = np.linalg.norm(p1 - p0) / dt
        else:
            vx, vy = traj.get_velocity(traj.last_frame)
            speed = np.hypot(vx, vy)
        self.roi_stats.append({
            'id': track.id,
            'airtime': airtime,
            'speed': speed,
            'detections': traj.length,
            'displacement': track.total_displacement,
            'fit_error': traj.fit_error,
            'positions': list(traj.positions),
            'positions_3d': list(traj.positions3d) if hasattr(traj, 'positions3d') else None,
            'frames': list(traj.frames),
            'coeffs_x': traj.coeffs_x.copy() if traj.coeffs_x is not None else None,
            'coeffs_y': traj.coeffs_y.copy() if traj.coeffs_y is not None else None,
            'coeffs_X': traj.coeffs_X.copy() if getattr(traj, 'coeffs_X', None) is not None else None,
            'coeffs_Y': traj.coeffs_Y.copy() if getattr(traj, 'coeffs_Y', None) is not None else None,
            'coeffs_Z': traj.coeffs_Z.copy() if getattr(traj, 'coeffs_Z', None) is not None else None,
            'color': track.color,
        })

    def _is_valid_flight(self, track):
        """A track is worth visualizing only once it has enough points,
        meaningful displacement, and a reasonable ballistic fit."""
        traj = track.trajectory
        if traj.length < self.min_track_length:
            return False
        if track.total_displacement < self.min_displacement:
            return False
        if traj.fit_error > self.max_fit_error:
            return False
        return True

    def _is_spurious(self, track):
        """Reject tracks that don't follow a reasonable ballistic trajectory."""
        traj = track.trajectory
        # Don't filter tracks that are too young to judge
        if traj.length < self.min_track_length + 1:
            return False
        # High fit error means detections don't lie on a parabola
        if traj.fit_error > self.max_fit_error:
            return True
        # No meaningful movement — likely a static background blob
        if track.total_displacement < self.min_displacement:
            return True
        return False

    # ---------------- ZED helpers -----------------
    def enable_zed(self, cam, runtime):
        """Initialize ZED-related state on the tracker so it can provide 3D points.

        cam: sl.Camera instance
        runtime: sl.RuntimeParameters used for grab()
        """
        if not _has_zed or cam is None:
            return
        try:
            self.zed_cam = cam
            self._zed_runtime = runtime
            self._zed_pc_mat = sl.Mat()
            
            info = cam.get_camera_information()
            intr = None
            try:
                cam_conf = getattr(info, 'camera_configuration', None)
                calib = getattr(cam_conf, 'calibration_parameters', None)
                left = getattr(calib, 'left_cam', None)
                fx = getattr(left, 'fx', None)
                fy = getattr(left, 'fy', None)
                cx = getattr(left, 'cx', None)
                cy = getattr(left, 'cy', None)
                if fx is not None and fy is not None and cx is not None and cy is not None:
                    intr = dict(fx=float(fx), fy=float(fy), cx=float(cx), cy=float(cy))
            except Exception:
                intr = None
            self.zed_intrinsics = intr
        except Exception:
            self.zed_cam = None
            self._zed_runtime = None
            self._zed_pc_mat = None
            self.zed_intrinsics = None

    def _get_point3d(self, u, v):
        """Return (X,Y,Z) in meters in camera coordinates for pixel (u,v) using the latest point cloud.

        Returns None if no valid depth available.
        """
        if not _has_zed or self.zed_cam is None or getattr(self, '_zed_pc_mat', None) is None:
            return None
        try:
            # read point cloud at pixel (u,v)
            res = self._zed_pc_mat.get_value(int(round(u)), int(round(v)))
            if isinstance(res, tuple) and len(res) == 2:
                err, point3D = res
            else:
                err = sl.ERROR_CODE.SUCCESS
                point3D = res
                
            if err != sl.ERROR_CODE.SUCCESS or point3D is None:
                return None
                
            x, y, z = point3D[0], point3D[1], point3D[2]
            
            import math
            distance = math.sqrt(x*x + y*y + z*z)
            if np.isnan(distance) or np.isinf(distance) or distance <= 0:
                return None
                
            # Point cloud returns values in MILLIMETERS since we configured the init_params to MILLIMETER
            # Convert values manually back to Meters so the rest of the app's physics scaling remains stable
            X = float(x) / 1000.0
            Y = float(y) / 1000.0
            Z = float(z) / 1000.0
            
            return (X, Y, Z)
        except Exception:
            return None

    # ------------------------------------------------------------------ #
    #  Visualization
    # ------------------------------------------------------------------ #

    def _visualize(self, frame, detections):
        vis = frame.copy()
        
        # Update World System and Draw AprilTags/Plane
        vis = self.world_system.update_pose(vis, getattr(self, 'zed_intrinsics', None))

        # Current-frame raw detections in green
        for (cx, cy, r) in detections:
            cv2.circle(vis, (int(cx), int(cy)), int(r), (0, 255, 0), 2)

        # Draw only tracks that have proven to be real ballistic flights
        for tr in self.tracks:
            traj = tr.trajectory
            if not self._is_valid_flight(tr):
                continue
            color = tr.color

            # Draw past trajectory as polyline
            if traj.length >= 2:
                pts = np.array([(int(p[0]), int(p[1])) for p in traj.positions], np.int32)
                cv2.polylines(vis, [pts], isClosed=False, color=color, thickness=2,
                              lineType=cv2.LINE_AA)

            # Draw fitted curve over the frame range (smoother than raw points)
            if traj.length >= 3:
                f0, f1 = traj.first_frame, traj.last_frame
                curve_pts = []
                for fi in np.linspace(f0, f1, num=max(30, (f1 - f0) * 3)):
                    p = traj.predict(fi)
                    if p:
                        curve_pts.append((int(p[0]), int(p[1])))
                if len(curve_pts) > 1:
                    cv2.polylines(vis, [np.array(curve_pts, np.int32)], False,
                                  color, 3, cv2.LINE_AA)

            # Future prediction (dashed)
            if traj.length >= 2:
                h, w = frame.shape[:2]
                prev = traj.predict(self.frame_count)
                for dt in range(1, 40):
                    nxt = traj.predict(self.frame_count + dt)
                    if nxt is None or not (-50 <= nxt[0] <= w + 50 and -50 <= nxt[1] <= h + 50):
                        break
                    if dt % 3 == 0 and prev is not None:
                        cv2.line(vis, (int(prev[0]), int(prev[1])),
                                 (int(nxt[0]), int(nxt[1])), color, 2, cv2.LINE_AA)
                    prev = nxt

            # Current predicted position marker + label
            pred = tr.predict(self.frame_count)
            if pred:
                cx, cy = int(pred[0]), int(pred[1])
                cv2.circle(vis, (cx, cy), int(tr.avg_radius), color, 2)

                vx, vy = traj.get_velocity(self.frame_count)
                speed = np.hypot(vx, vy)

                # Velocity arrow
                if speed > 2:
                    scale = min(3.0, 50 / (speed + 1))
                    cv2.arrowedLine(vis, (cx, cy),
                                    (int(cx + vx * scale), int(cy + vy * scale)),
                                    color, 2, tipLength=0.3, line_type=cv2.LINE_AA)

                air_t = tr.airtime(self.frame_count, self.fps)
                label = f"ID:{tr.id} t={air_t:.2f}s"
                cv2.putText(vis, label, (cx - 40, cy - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                label2 = f"v={speed:.0f} err={traj.fit_error:.1f}"
                cv2.putText(vis, label2, (cx - 40, cy - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)

            # --- Check plane intersection ---
            if hasattr(traj, 'positions3d') and len(traj.positions3d) >= 2:
                p_cam_prev = traj.positions3d[-2]
                p_cam_new = traj.positions3d[-1]
                if p_cam_prev is not None and p_cam_new is not None:
                    p_w_prev = self.world_system.cam_to_world(p_cam_prev)
                    p_w_new = self.world_system.cam_to_world(p_cam_new)
                    intersect = self.world_system.check_plane_intersection(p_w_prev, p_w_new)
                    if intersect:
                        self.intersections.append({'id': tr.id, 'point': intersect})
                        print(f"BINGO! Ball ID #{tr.id} intercepted target plane at World XYZ: {intersect[0]:.2f}, {intersect[1]:.2f}, {intersect[2]:.2f}")

        # Draw ROI rectangle
        if self.roi is not None:
            rx, ry, rw, rh = self.roi
            cv2.rectangle(vis, (rx, ry), (rx + rw, ry + rh), (0, 255, 255), 2)
            cv2.putText(vis, "ROI", (rx + 4, ry + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # HUD
        n_confirmed = sum(1 for t in self.tracks if self._is_valid_flight(t))
        info = (f"Frame:{self.frame_count} | "
                f"Tracks:{n_confirmed} ({len(self.tracks)} total) | "
                f"Detections:{len(detections)}")
        cv2.putText(vis, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

        # ROI stats panel
        if self.roi is not None and self.roi_stats:
            h_frame = vis.shape[0]
            panel_x = 10
            panel_y = h_frame - 20 - len(self.roi_stats) * 20 - 25
            cv2.putText(vis, f"ROI balls: {len(self.roi_stats)}", (panel_x, panel_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            for i, s in enumerate(self.roi_stats):
                y = panel_y + 22 + i * 20
                line = (f"#{s['id']}  t={s['airtime']:.2f}s  "
                        f"v={s['speed']:.0f}  disp={s['displacement']:.0f}px")
                cv2.putText(vis, line, (panel_x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)

        return vis

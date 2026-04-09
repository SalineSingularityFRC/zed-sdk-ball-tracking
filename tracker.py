import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from models import Track
from zed_utils import sl, _has_zed

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
        self._zed_depth_mat = None
        self.zed_intrinsics = None  # dict with fx,fy,cx,cy if available

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
                        if self._zed_depth_mat is not None:
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
                    if self._zed_depth_mat is not None:
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
            self._zed_depth_mat = sl.Mat()
            # Try to read intrinsics
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
            self._zed_depth_mat = None
            self.zed_intrinsics = None

    def _get_point3d(self, u, v):
        """Return (X,Y,Z) in meters in camera coordinates for pixel (u,v) using the latest depth mat.

        Returns None if no valid depth or intrinsics available.
        """
        if not _has_zed or self.zed_cam is None or self._zed_depth_mat is None:
            return None
        try:
            # read depth at pixel (u,v). depth is in meters
            z = self._zed_depth_mat.get_value(int(round(u)), int(round(v)))
            if z is None:
                return None
            z = float(z)
            if np.isnan(z) or z <= 0:
                return None
            intr = self.zed_intrinsics
            if intr is None:
                return None
            fx = intr['fx']; fy = intr['fy']; cx = intr['cx']; cy = intr['cy']
            X = (u - cx) * z / fx
            Y = (v - cy) * z / fy
            Z = z
            return (float(X), float(Y), float(Z))
        except Exception:
            return None

    # ------------------------------------------------------------------ #
    #  Visualization
    # ------------------------------------------------------------------ #

    def _visualize(self, frame, detections):
        vis = frame.copy()

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

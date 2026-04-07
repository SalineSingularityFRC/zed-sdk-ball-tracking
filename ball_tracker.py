import cv2
import argparse
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment

# Try to import ZED SDK (pyzed). This is optional; fall back to OpenCV camera if unavailable.
try:
    import pyzed.sl as sl
    _has_zed = True
except Exception:
    sl = None
    _has_zed = False


class BallisticTrajectory:
    """
    Incrementally-fitted ballistic trajectory.
    Models x(t) as linear and y(t) as quadratic (gravity).
    Refits on every new detection for accuracy.
    """

    def __init__(self):
        self.positions = []  # list of (x, y)
        self.frames = []     # list of frame indices
        self.coeffs_x = None
        self.coeffs_y = None
        self.fit_error = 0.0

    def add(self, x, y, frame_idx):
        self.positions.append((x, y))
        self.frames.append(frame_idx)
        self._refit()

    def _refit(self):
        n = len(self.positions)
        if n < 2:
            # With 1 point, no fit possible — predict stays at that point
            self.coeffs_x = None
            self.coeffs_y = None
            return

        pts = np.array(self.positions)
        t = np.array(self.frames, dtype=float)

        # x: linear (deg 1), but use deg 0 if only 2 points and they're close in time
        deg_x = min(1, n - 1)
        self.coeffs_x = np.polyfit(t, pts[:, 0], deg=deg_x)

        # y: quadratic for gravity, but degrade gracefully
        deg_y = min(2, n - 1)
        self.coeffs_y = np.polyfit(t, pts[:, 1], deg=deg_y)

        # Compute fit error
        pred_x = np.polyval(self.coeffs_x, t)
        pred_y = np.polyval(self.coeffs_y, t)
        errors = np.sqrt((pts[:, 0] - pred_x) ** 2 + (pts[:, 1] - pred_y) ** 2)
        self.fit_error = np.mean(errors)

    def predict(self, frame_idx):
        """Predict position at frame_idx."""
        if len(self.positions) == 0:
            return None
        if len(self.positions) == 1:
            # Can't extrapolate, return last known position
            return self.positions[0]
        t = float(frame_idx)
        x = np.polyval(self.coeffs_x, t)
        y = np.polyval(self.coeffs_y, t)
        return (x, y)

    def get_velocity(self, frame_idx):
        """Get velocity at frame_idx from polynomial derivatives."""
        if self.coeffs_x is None or self.coeffs_y is None:
            return (0.0, 0.0)
        # derivative of polynomial
        dx = np.polyder(self.coeffs_x)
        dy = np.polyder(self.coeffs_y)
        vx = np.polyval(dx, float(frame_idx))
        vy = np.polyval(dy, float(frame_idx))
        return (vx, vy)

    @property
    def last_frame(self):
        return self.frames[-1] if self.frames else -1

    @property
    def first_frame(self):
        return self.frames[0] if self.frames else -1

    @property
    def length(self):
        return len(self.positions)


class Track:
    """A single tracked ball with its trajectory and metadata."""

    _next_id = 0

    def __init__(self, x, y, radius, frame_idx):
        self.id = Track._next_id
        Track._next_id += 1
        self.trajectory = BallisticTrajectory()
        self.trajectory.add(x, y, frame_idx)
        self.radii = [radius]
        self.missed_frames = 0  # consecutive frames without a detection
        self.color = tuple(map(int, np.random.randint(100, 255, 3)))

    def predict(self, frame_idx):
        return self.trajectory.predict(frame_idx)

    def update(self, x, y, radius, frame_idx):
        self.trajectory.add(x, y, frame_idx)
        self.radii.append(radius)
        self.missed_frames = 0

    def mark_missed(self):
        self.missed_frames += 1

    @property
    def avg_radius(self):
        return np.mean(self.radii[-10:])  # recent average

    def airtime(self, current_frame, fps):
        """Airtime in seconds from first detection to current frame."""
        return (current_frame - self.trajectory.first_frame) / fps

    @property
    def total_displacement(self):
        """Euclidean distance from first to last detection."""
        if self.trajectory.length < 2:
            return 0.0
        p0 = self.trajectory.positions[0]
        p1 = self.trajectory.positions[-1]
        return np.hypot(p1[0] - p0[0], p1[1] - p0[1])


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
        """
        Args:
            lower_hsv/upper_hsv: Color segmentation thresholds.
            min_radius/max_radius: Ball size constraints in pixels.
            gate_radius: Max distance (px) between prediction and detection for association.
            max_missed: Drop track after this many consecutive missed frames.
            min_track_length: Minimum detections before a track is considered confirmed.
            min_circularity: Minimum circularity for contour detection.
            fps: Video frame rate, used to compute airtime.
            max_fit_error: Max average fit error (px) before a track is rejected as non-ballistic.
            min_displacement: Min total displacement (px) for a track to be considered real motion.
            roi: (x, y, w, h) region of interest, or None to skip ROI filtering.
        """
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
                    self.tracks[i].update(dx, dy, dr, self.frame_count)
                    assigned_det.add(j)
                    assigned_track.add(i)

        # --- Mark unassigned tracks as missed ---
        for i, tr in enumerate(self.tracks):
            if i not in assigned_track:
                tr.mark_missed()

        # --- Create new tracks from unassigned detections ---
        for j, (dx, dy, dr) in enumerate(detections):
            if j not in assigned_det:
                self.tracks.append(Track(dx, dy, dr, self.frame_count))

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

def run_calibration(video_path=None, image_path=None, camera_index=0):
    def nothing(x):
        pass

    if image_path:
        if not Path(image_path).exists():
            print(f"Error: Image file '{image_path}' not found"); return
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Error: Could not load image '{image_path}'"); return
        cap = None
    elif video_path:
        if not Path(video_path).exists():
            print(f"Error: Video file '{video_path}' not found"); return
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video '{video_path}'"); return
    else:
        # Prefer ZED camera when available
        use_zed = False
        cap = None
        if _has_zed:
            try:
                cam = sl.Camera()
                init = sl.InitParameters()
                # Use default init params; user can tune if needed
                status = cam.open(init)
                if status == sl.ERROR_CODE.SUCCESS:
                    runtime = sl.RuntimeParameters()
                    mat = sl.Mat()
                    # small warm-up and single frame grab
                    if cam.grab(runtime) == sl.ERROR_CODE.SUCCESS:
                        cam.retrieve_image(mat, sl.VIEW.LEFT)
                        frame = mat.get_data()
                        # Convert RGBA (ZED) to BGR for OpenCV if needed
                        if frame is not None and frame.ndim == 3 and frame.shape[2] == 4:
                            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
                        use_zed = True
                        cap = cam
                    else:
                        cam.close()
                        cap = None
                else:
                    # failed to open ZED, fall back to cv2
                    cap = None
            except Exception:
                cap = None

        if cap is None:
            cap = cv2.VideoCapture(camera_index)
            if not cap.isOpened():
                print(f"Error: Could not open camera {camera_index}"); return
            import time; time.sleep(0.5)
            ret, frame = cap.read()
            if not ret:
                print("Error: Camera opened but cannot read frames");
                # cleanup
                if hasattr(cap, 'release'):
                    cap.release()
                return

    window_name = 'HSV Calibration Tool'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    for name, mx, default in [('H Min', 179, 0), ('H Max', 179, 179),
                               ('S Min', 255, 0), ('S Max', 255, 255),
                               ('V Min', 255, 0), ('V Max', 255, 255)]:
        cv2.createTrackbar(name, window_name, default, mx, nothing)

    print("\nHSV Calibration — 'q' quit, 's' save, 'r' reset")

    try:
        while True:
            if cap is not None:
                # If using ZED camera instance
                if _has_zed and isinstance(cap, sl.Camera):
                    if cap.grab(runtime) != sl.ERROR_CODE.SUCCESS:
                        break
                    mat = sl.Mat()
                    cap.retrieve_image(mat, sl.VIEW.LEFT)
                    new_frame = mat.get_data()
                    if new_frame is None:
                        break
                    if new_frame.ndim == 3 and new_frame.shape[2] == 4:
                        new_frame = cv2.cvtColor(new_frame, cv2.COLOR_RGBA2RGB)
                    frame = new_frame
                else:
                    ret, new_frame = cap.read()
                    if not ret:
                        if video_path:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0); continue
                        else:
                            break
                    frame = new_frame

            vals = {n: cv2.getTrackbarPos(n, window_name)
                    for n in ['H Min', 'H Max', 'S Min', 'S Max', 'V Min', 'V Max']}
            lower = np.array([vals['H Min'], vals['S Min'], vals['V Min']])
            upper = np.array([vals['H Max'], vals['S Max'], vals['V Max']])

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower, upper)
            result = cv2.bitwise_and(frame, frame, mask=mask)

            dh = 400
            h, w = frame.shape[:2]
            dw = int(w * dh / h)
            row = np.hstack([cv2.resize(frame, (dw, dh)),
                             cv2.resize(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), (dw, dh)),
                             cv2.resize(result, (dw, dh))])
            cv2.imshow(window_name, row)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print(f"\nLower HSV: [{vals['H Min']}, {vals['S Min']}, {vals['V Min']}]")
                print(f"Upper HSV: [{vals['H Max']}, {vals['S Max']}, {vals['V Max']}]")
                break
            elif key == ord('s'):
                with open('hsv_config.txt', 'w') as f:
                    f.write(f"lower_hsv = [{vals['H Min']}, {vals['S Min']}, {vals['V Min']}]\n")
                    f.write(f"upper_hsv = [{vals['H Max']}, {vals['S Max']}, {vals['V Max']}]\n")
                print("Saved to hsv_config.txt")
            elif key == ord('r'):
                for n, v in [('H Min', 0), ('H Max', 179), ('S Min', 0),
                             ('S Max', 255), ('V Min', 0), ('V Max', 255)]:
                    cv2.setTrackbarPos(n, window_name, v)
    finally:
        # Cleanup capture
        if cap is not None:
            try:
                if _has_zed and isinstance(cap, sl.Camera):
                    cap.close()
                elif hasattr(cap, 'release'):
                    cap.release()
            except Exception:
                pass
        cv2.destroyAllWindows()


def load_hsv_config():
    config_path = Path('hsv_config.txt')
    if not config_path.exists():
        return None, None
    try:
        lower, upper = None, None
        for line in config_path.read_text().splitlines():
            line = line.strip()
            if line.startswith('lower_hsv'):
                lower = [int(x.strip()) for x in line.split('=')[1].strip().strip('[]').split(',')]
            elif line.startswith('upper_hsv'):
                upper = [int(x.strip()) for x in line.split('=')[1].strip().strip('[]').split(',')]
        if lower and upper:
            print(f"Loaded HSV config: lower={lower} upper={upper}")
            return lower, upper
    except Exception as e:
        print(f"Warning: could not load HSV config: {e}")
    return None, None


def _show_roi_summary(background, tracker):
    """Show a final image with all ROI trajectories overlaid. Press any key to close."""
    vis = background.copy()
    # Dim the background so trajectories stand out
    vis = (vis * 0.4).astype(np.uint8)

    # Draw ROI
    rx, ry, rw, rh = tracker.roi
    cv2.rectangle(vis, (rx, ry), (rx + rw, ry + rh), (0, 255, 255), 2)

    for s in tracker.roi_stats:
        color = s['color']
        positions = s['positions']
        coeffs_x = s['coeffs_x']
        coeffs_y = s['coeffs_y']
        frames = s['frames']

        # Draw raw detection points
        for px, py in positions:
            cv2.circle(vis, (int(px), int(py)), 4, color, -1, cv2.LINE_AA)

        # Draw fitted curve
        if coeffs_x is not None and coeffs_y is not None and len(frames) >= 3:
            f0, f1 = frames[0], frames[-1]
            curve_pts = []
            for fi in np.linspace(f0, f1, num=max(50, (f1 - f0) * 3)):
                x = np.polyval(coeffs_x, fi)
                y = np.polyval(coeffs_y, fi)
                curve_pts.append((int(x), int(y)))
            if len(curve_pts) > 1:
                cv2.polylines(vis, [np.array(curve_pts, np.int32)], False,
                              color, 2, cv2.LINE_AA)

        # Label at the midpoint of the trajectory
        mid = positions[len(positions) // 2]
        label = f"#{s['id']} {s['airtime']:.2f}s v={s['speed']:.0f}"
        cv2.putText(vis, label, (int(mid[0]) + 8, int(mid[1]) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    # Summary text at top
    cv2.putText(vis, f"ROI Summary: {len(tracker.roi_stats)} trajectories",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    airtimes = [s['airtime'] for s in tracker.roi_stats]
    speeds = [s['speed'] for s in tracker.roi_stats]
    cv2.putText(vis, f"Airtime avg={np.mean(airtimes):.2f}s  Speed avg={np.mean(speeds):.0f}",
                (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow('ROI Summary', vis)
    print("\nShowing ROI summary — press any key to close.")
    cv2.waitKey(0)
    cv2.destroyWindow('ROI Summary')


def run_tracker(video_path=None, camera_index=0, start_from=0.0, no_roi=False):
    lower_hsv, upper_hsv = load_hsv_config()

    kwargs = {}
    if lower_hsv and upper_hsv:
        kwargs['lower_hsv'] = lower_hsv
        kwargs['upper_hsv'] = upper_hsv
    else:
        print("Using default HSV values (run with --calibrate to tune)")

    use_zed = False
    cap = None
    if video_path:
        if not Path(video_path).exists():
            print(f"Error: Video file '{video_path}' not found"); return
        cap = cv2.VideoCapture(video_path)
    else:
        # Prefer ZED camera when available
        if _has_zed:
            try:
                cam = sl.Camera()
                init = sl.InitParameters()
                status = cam.open(init)
                if status == sl.ERROR_CODE.SUCCESS:
                    runtime = sl.RuntimeParameters()
                    cap = cam
                    use_zed = True
                else:
                    cam.close()
                    cap = None
            except Exception:
                cap = None

        if cap is None:
            cap = cv2.VideoCapture(camera_index)

    # Validate capture
    try:
        opened = cap.isOpened() if not (use_zed and isinstance(cap, sl.Camera)) else True
    except Exception:
        opened = False
    if not opened:
        print("Error: Could not open video source"); return

    # Determine fps
    fps = None
    if use_zed and isinstance(cap, sl.Camera):
        try:
            info = cap.get_camera_information()
            fps = getattr(info.camera_configuration, 'fps', None)
        except Exception:
            fps = None
    else:
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
        except Exception:
            fps = None

    if not fps or fps <= 0:
        fps = 30.0
    kwargs['fps'] = fps

    if start_from > 0 and video_path:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_from * 1000.0)
        actual = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        print(f"Seeked to {actual:.1f}s (requested {start_from:.1f}s)")

    # --- ROI selection from first frame ---
    # Read first frame (handle ZED differently)
    if use_zed and isinstance(cap, sl.Camera):
        mat = sl.Mat()
        if cap.grab(runtime) != sl.ERROR_CODE.SUCCESS:
            print("Error: Could not read first frame from ZED"); cap.close(); return
        cap.retrieve_image(mat, sl.VIEW.LEFT)
        first_frame = mat.get_data()
        if first_frame is None:
            print("Error: Could not read first frame from ZED"); cap.close(); return
        if first_frame.ndim == 3 and first_frame.shape[2] == 4:
            first_frame = cv2.cvtColor(first_frame, cv2.COLOR_RGBA2RGB)
    else:
        ret, first_frame = cap.read()
        if not ret:
            print("Error: Could not read first frame");
            if hasattr(cap, 'release'):
                cap.release()
            return

    if not no_roi:
        print("Select ROI and press ENTER/SPACE. Press 'c' to cancel (no ROI).")
        roi = cv2.selectROI("Select ROI", first_frame, showCrosshair=True, fromCenter=False)
        cv2.destroyWindow("Select ROI")
        if roi[2] > 0 and roi[3] > 0:
            kwargs['roi'] = tuple(int(v) for v in roi)
            print(f"ROI set: x={roi[0]} y={roi[1]} w={roi[2]} h={roi[3]}")
        else:
            print("No ROI selected, tracking without ROI")

    tracker = BallTracker(**kwargs)

    # Process the first frame we already read
    mask = tracker.segment(first_frame)
    vis = tracker.track(first_frame, mask)
    cv2.imshow('Ball Tracker', vis)
    cv2.imshow('Mask', mask)

    print("Press 'q' to quit")
    try:
        while True:
            # Read frame depending on capture type
            if use_zed and isinstance(cap, sl.Camera):
                mat = sl.Mat()
                if cap.grab(runtime) != sl.ERROR_CODE.SUCCESS:
                    print("End of ZED stream"); break
                cap.retrieve_image(mat, sl.VIEW.LEFT)
                frame = mat.get_data()
                if frame is None:
                    print("End of ZED stream"); break
                if frame.ndim == 3 and frame.shape[2] == 4:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
            else:
                ret, frame = cap.read()
                if not ret:
                    print("End of video"); break

            mask = tracker.segment(frame)
            vis = tracker.track(frame, mask)

            cv2.imshow('Ball Tracker', vis)
            cv2.imshow('Mask', mask)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # Cleanup capture
        if cap is not None:
            try:
                if use_zed and isinstance(cap, sl.Camera):
                    cap.close()
                elif hasattr(cap, 'release'):
                    cap.release()
            except Exception:
                pass
        cv2.destroyAllWindows()

        # Flush any still-active tracks into ROI stats
        if tracker.roi is not None:
            for t in tracker.tracks:
                tracker._capture_roi_stats(t)

        # Print ROI summary and show final trajectory visualization
        if tracker.roi_stats:
            print(f"\n{'='*50}")
            print(f"ROI Summary: {len(tracker.roi_stats)} balls detected")
            print(f"{'='*50}")
            print(f"{'ID':>4}  {'Airtime':>8}  {'Speed':>6}  {'Disp':>7}  {'Pts':>4}  {'Err':>5}")
            for s in tracker.roi_stats:
                print(f"#{s['id']:>3}  {s['airtime']:>7.2f}s  {s['speed']:>6.0f}  "
                      f"{s['displacement']:>6.0f}px  {s['detections']:>4}  {s['fit_error']:>5.1f}")
            airtimes = [s['airtime'] for s in tracker.roi_stats]
            speeds = [s['speed'] for s in tracker.roi_stats]
            print(f"\nAirtime  avg={np.mean(airtimes):.2f}s  "
                  f"min={min(airtimes):.2f}s  max={max(airtimes):.2f}s")
            print(f"Speed    avg={np.mean(speeds):.0f}  "
                  f"min={min(speeds):.0f}  max={max(speeds):.0f}")

            _show_roi_summary(first_frame, tracker)


def main():
    parser = argparse.ArgumentParser(description='Multi-ball ballistic tracker')
    parser.add_argument('--calibrate', action='store_true')
    parser.add_argument('--video', type=str)
    parser.add_argument('--image', type=str)
    parser.add_argument('--camera', type=int, default=1)
    parser.add_argument('--start-from', type=float, default=0.0,
                        help='Skip to this many seconds into the video')
    parser.add_argument('--no-roi', action='store_true',
                        help='Skip ROI selection')
    args = parser.parse_args()

    if args.calibrate:
        run_calibration(video_path=args.video, image_path=args.image, camera_index=args.camera)
    else:
        run_tracker(video_path=args.video, camera_index=args.camera,
                    start_from=args.start_from, no_roi=args.no_roi)


if __name__ == '__main__':
    main()
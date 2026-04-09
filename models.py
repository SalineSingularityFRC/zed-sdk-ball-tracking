import numpy as np

class BallisticTrajectory:
    """
    Incrementally-fitted ballistic trajectory.
    Models x(t) as linear and y(t) as quadratic (gravity).
    Refits on every new detection for accuracy.
    """

    def __init__(self):
        self.positions = []  # list of (x, y)
        self.positions3d = []  # list of (X, Y, Z) in meters (camera coordinates), optional
        self.frames = []     # list of frame indices
        self.coeffs_x = None
        self.coeffs_y = None
        self.coeffs_X = None  # 3D X polynomial
        self.coeffs_Y = None  # 3D Y polynomial
        self.coeffs_Z = None  # 3D Z polynomial
        self.fit_error = 0.0

    def add(self, x, y, frame_idx, pos3d=None):
        self.positions.append((x, y))
        if pos3d is not None:
            self.positions3d.append(tuple(pos3d))
        else:
            # keep lists aligned by filling with None if 3D unavailable
            self.positions3d.append(None)
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

        # 3D fit
        valid_3d = [(float(f), p[0], p[1], p[2]) for f, p in zip(self.frames, self.positions3d) if p is not None]
        n3d = len(valid_3d)
        if n3d >= 2:
            t3 = np.array([v[0] for v in valid_3d])
            X = np.array([v[1] for v in valid_3d])
            Y = np.array([v[2] for v in valid_3d])
            Z = np.array([v[3] for v in valid_3d])

            deg_xz = min(1, n3d - 1)
            deg_y = min(2, n3d - 1)

            self.coeffs_X = np.polyfit(t3, X, deg=deg_xz)
            self.coeffs_Y = np.polyfit(t3, Y, deg=deg_y)
            self.coeffs_Z = np.polyfit(t3, Z, deg=deg_xz)
        else:
            self.coeffs_X = None
            self.coeffs_Y = None
            self.coeffs_Z = None

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
        # pos3d may be provided later by caller; default to None
        self.trajectory.add(x, y, frame_idx, pos3d=None)
        self.radii = [radius]
        self.missed_frames = 0  # consecutive frames without a detection
        self.color = tuple(map(int, np.random.randint(100, 255, 3)))

    def predict(self, frame_idx):
        return self.trajectory.predict(frame_idx)

    def update(self, x, y, radius, frame_idx, pos3d=None):
        self.trajectory.add(x, y, frame_idx, pos3d=pos3d)
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
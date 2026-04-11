"""Project tracker roi_stats entries into the field frame using the
localizer's world pose. Shared by visualization and shot logging so the
two paths can never disagree about where a track lives in 3D."""

import numpy as np
from gtsam import Point3


def world_points_for_stat(stat: dict, world_pose, n_samples: int = 400,
                          extrapolate: bool = True) -> list[tuple[float, float, float]] | None:
    """Sample the polynomial fit of an roi_stats entry along its frame range
    and transform each sample into the field frame. Returns None if the
    stat doesn't have enough information to project."""
    cX = stat.get("coeffs_X")
    cY = stat.get("coeffs_Y")
    cZ = stat.get("coeffs_Z")
    frames = stat.get("frames")
    if cX is None or cY is None or cZ is None or not frames or len(frames) < 2:
        return None

    f0, f1 = frames[0], frames[-1]
    span = max(f1 - f0, 1)
    f_end = f1 + int(span) if extrapolate else f1
    t_vals = np.linspace(f0, f_end, n_samples)
    curve_X = np.polyval(cX, t_vals)
    curve_Y = np.polyval(cY, t_vals)
    curve_Z = np.polyval(cZ, t_vals)

    pts: list[tuple[float, float, float]] = []
    for X, Y, Z in zip(curve_X, curve_Y, curve_Z):
        wp = world_pose.transformFrom(Point3(float(X), float(Y), float(Z)))
        pts.append((float(wp[0]), float(wp[1]), float(wp[2])))
    return pts if len(pts) >= 2 else None

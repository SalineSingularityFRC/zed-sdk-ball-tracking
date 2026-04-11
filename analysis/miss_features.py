"""Pure geometry helpers for turning robot/turret state + miss point into
features for the miss-vs-state regression. No I/O, no ZED, no NT."""

import math
from dataclasses import dataclass


def _wrap(theta: float) -> float:
    return math.atan2(math.sin(theta), math.cos(theta))


@dataclass
class TurretFrame:
    """Turret pose in the field frame."""
    x: float
    y: float
    z: float
    theta: float  # absolute heading the turret is currently aiming, in field frame


def turret_robot_frame(robot_x: float, robot_y: float, robot_z: float,
                       robot_theta: float,
                       offset_x: float, offset_y: float, offset_z: float,
                       turret_angle: float) -> TurretFrame:
    """Compose the robot's chassis pose with the turret mounting offset and
    its current articulation angle. The offset is given in the robot's body
    frame and is rotated by the robot's yaw before being added to the robot
    translation. The turret angle is reported relative to the chassis, so the
    world-frame heading is robot_theta + turret_angle."""
    c = math.cos(robot_theta)
    s = math.sin(robot_theta)
    tx = robot_x + c * offset_x - s * offset_y
    ty = robot_y + s * offset_x + c * offset_y
    tz = robot_z + offset_z
    return TurretFrame(x=tx, y=ty, z=tz, theta=_wrap(robot_theta + turret_angle))


def turret_relative_polar(turret: TurretFrame,
                          target_x: float, target_y: float) -> tuple[float, float]:
    """Return (range, bearing): horizontal distance from the turret to the
    target, and the angle from the turret's aim direction to the target."""
    dx = target_x - turret.x
    dy = target_y - turret.y
    rng = math.hypot(dx, dy)
    bearing = _wrap(math.atan2(dy, dx) - turret.theta)
    return rng, bearing


def decompose_velocity(vel_x: float, vel_y: float,
                       turret: TurretFrame,
                       target_x: float, target_y: float) -> tuple[float, float]:
    """Project (vel_x, vel_y) onto the turret->target line and its perpendicular.
    Returns (v_radial, v_tangential). v_radial is positive when moving toward
    the target."""
    dx = target_x - turret.x
    dy = target_y - turret.y
    n = math.hypot(dx, dy)
    if n < 1e-9:
        return 0.0, 0.0
    ux, uy = dx / n, dy / n
    v_rad = vel_x * ux + vel_y * uy
    v_tan = -vel_x * uy + vel_y * ux
    return v_rad, v_tan


def miss_in_polar(miss_x: float, miss_y: float,
                  turret: TurretFrame,
                  target_x: float, target_y: float) -> tuple[float, float]:
    """Express the miss vector (impact point - target) in polar coordinates
    measured relative to the turret->target direction. miss_theta = 0 means
    the ball landed past the target along the line of fire; positive values
    are to the left of that line."""
    dx = miss_x - target_x
    dy = miss_y - target_y
    miss_r = math.hypot(dx, dy)
    aim_dx = target_x - turret.x
    aim_dy = target_y - turret.y
    if math.hypot(aim_dx, aim_dy) < 1e-9:
        return miss_r, 0.0
    aim_theta = math.atan2(aim_dy, aim_dx)
    miss_theta = _wrap(math.atan2(dy, dx) - aim_theta)
    return miss_r, miss_theta

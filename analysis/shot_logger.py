import csv
from pathlib import Path

FIELDS = [
    "shot_id", "svo", "frame_start", "frame_cross",
    "robot_x", "robot_y", "robot_theta",
    "vel_x", "vel_y", "omega", "turret_angle",
    "turret_world_x", "turret_world_y", "turret_world_theta",
    "target_x", "target_y", "target_z",
    "range_to_target", "bearing_to_target",
    "v_radial", "v_tangential",
    "miss_x", "miss_y", "miss_r", "miss_theta",
]


class ShotLogger:
    """Append-only CSV writer for one row per ball that crosses the target plane."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._needs_header = not self.path.exists() or self.path.stat().st_size == 0

    def append(self, row: dict) -> None:
        with open(self.path, "a", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=FIELDS, extrasaction="ignore")
            if self._needs_header:
                writer.writeheader()
                self._needs_header = False
            writer.writerow({k: row.get(k) for k in FIELDS})

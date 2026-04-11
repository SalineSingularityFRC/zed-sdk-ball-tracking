import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class NTConfig:
    server: str = "localhost"
    table: str = "/SmartDashboard"
    keys: dict = field(default_factory=dict)


@dataclass
class TurretConfig:
    offset_x: float = 0.0
    offset_y: float = 0.0
    offset_z: float = 0.0


@dataclass
class TargetOverride:
    x: Optional[float] = None
    y: Optional[float] = None
    z: Optional[float] = None

    @property
    def has_override(self) -> bool:
        return self.x is not None and self.y is not None and self.z is not None


@dataclass
class LoggingConfig:
    shots_csv: str = "shots.csv"


@dataclass
class Config:
    nt: NTConfig = field(default_factory=NTConfig)
    turret: TurretConfig = field(default_factory=TurretConfig)
    target: TargetOverride = field(default_factory=TargetOverride)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @classmethod
    def load(cls, path: str | Path = "config.json") -> "Config":
        p = Path(path)
        if not p.exists():
            print(f"[config] {p} not found, using defaults")
            return cls()
        raw = json.loads(p.read_text())
        nt_raw = raw.get("nt", {})
        turret_raw = raw.get("turret", {})
        target_raw = raw.get("target", {})
        logging_raw = raw.get("logging", {})
        return cls(
            nt=NTConfig(
                server=nt_raw.get("server", "localhost"),
                table=nt_raw.get("table", "/SmartDashboard"),
                keys=nt_raw.get("keys", {}),
            ),
            turret=TurretConfig(
                offset_x=float(turret_raw.get("offset_x", 0.0)),
                offset_y=float(turret_raw.get("offset_y", 0.0)),
                offset_z=float(turret_raw.get("offset_z", 0.0)),
            ),
            target=TargetOverride(
                x=target_raw.get("x"),
                y=target_raw.get("y"),
                z=target_raw.get("z"),
            ),
            logging=LoggingConfig(
                shots_csv=logging_raw.get("shots_csv", "shots.csv"),
            ),
        )

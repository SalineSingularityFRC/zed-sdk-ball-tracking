import bisect
import json
from pathlib import Path
from typing import Optional


class NTLog:
    """Offline reader for NTRecorder JSONL sidecar files.

    Indexed by SVO frame number; at(frame) returns the row for that frame
    or the nearest preceding row if the exact frame wasn't recorded.
    """

    def __init__(self, rows: list[dict]):
        self._rows = sorted(rows, key=lambda r: r.get("frame", 0))
        self._frames = [int(r.get("frame", 0)) for r in self._rows]

    @classmethod
    def load(cls, path: str | Path) -> Optional["NTLog"]:
        p = Path(path)
        if not p.exists():
            return None
        rows: list[dict] = []
        with open(p, "r") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        if not rows:
            return None
        return cls(rows)

    @property
    def has_data(self) -> bool:
        return bool(self._rows)

    def at(self, frame_index: int) -> Optional[dict]:
        if not self._rows:
            return None
        i = bisect.bisect_right(self._frames, int(frame_index)) - 1
        if i < 0:
            return None
        return self._rows[i]

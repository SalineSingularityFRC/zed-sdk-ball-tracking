import json
from pathlib import Path

from .nt_client import NTRobotClient


class NTRecorder:
    """Writes one JSONL row per SVO frame containing a NetworkTables snapshot."""

    def __init__(self, client: NTRobotClient, output_path: str | Path):
        self.client = client
        self.path = Path(output_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = open(self.path, "w")
        self._closed = False

    def record_frame(self, frame_index: int) -> None:
        if self._closed:
            return
        snap = self.client.snapshot()
        snap["frame"] = int(frame_index)
        self._fp.write(json.dumps(snap) + "\n")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._fp.flush()
            self._fp.close()
        except Exception:
            pass

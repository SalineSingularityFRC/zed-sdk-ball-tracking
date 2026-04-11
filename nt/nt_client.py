import time
from typing import Optional

from config import NTConfig


class NTRobotClient:
    """Thin wrapper over ntcore (NT4) with a pynetworktables fallback.

    Subscribes to the NT keys named in NTConfig.keys (a dict of logical
    name -> full NT path) and exposes snapshot() returning the latest
    value for every key.
    """

    def __init__(self, cfg: NTConfig):
        self.cfg = cfg
        self._backend = None
        self._entries: dict = {}
        self._connect()

    def _connect(self) -> None:
        try:
            import ntcore  # type: ignore

            inst = ntcore.NetworkTableInstance.getDefault()
            inst.startClient4("zed-tracker")
            inst.setServer(self.cfg.server)
            for name, path in self.cfg.keys.items():
                topic = inst.getDoubleTopic(path)
                self._entries[name] = topic.subscribe(float("nan"))
            self._backend = ("ntcore", inst)
            print(f"[nt] ntcore client started, server={self.cfg.server}")
            return
        except ImportError:
            pass
        except Exception as e:
            print(f"[nt] ntcore init failed: {e}")

        try:
            from networktables import NetworkTables  # type: ignore

            NetworkTables.initialize(server=self.cfg.server)
            for name, path in self.cfg.keys.items():
                self._entries[name] = NetworkTables.getEntry(path)
            self._backend = ("pynetworktables", NetworkTables)
            print(f"[nt] pynetworktables client started, server={self.cfg.server}")
        except Exception as e:
            print(f"[nt] no NT backend available: {e}")
            self._backend = None

    @property
    def connected(self) -> bool:
        if self._backend is None:
            return False
        kind, handle = self._backend
        if kind == "ntcore":
            return handle.isConnected()
        return handle.isConnected()

    def snapshot(self) -> dict:
        out: dict = {"t_wall": time.time(), "connected": self.connected}
        if self._backend is None:
            for name in self.cfg.keys:
                out[name] = None
            return out
        kind, _ = self._backend
        for name, sub in self._entries.items():
            try:
                if kind == "ntcore":
                    v = sub.get()
                else:
                    v = sub.getDouble(float("nan"))
                if v != v:  # NaN
                    out[name] = None
                else:
                    out[name] = float(v)
            except Exception:
                out[name] = None
        return out

    def close(self) -> None:
        if self._backend is None:
            return
        kind, handle = self._backend
        try:
            if kind == "ntcore":
                handle.stopClient()
            else:
                handle.shutdown()
        except Exception:
            pass

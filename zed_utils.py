try:
    import pyzed.sl as sl
    _has_zed = True
except Exception:
    sl = None
    _has_zed = False

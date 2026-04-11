# zed-sdk-ball-tracking

## Requirements

- Python 3.10+
- ZED SDK (install from https://www.stereolabs.com/developers/release/) — provides `pyzed.sl`
- `pyzed` Python bindings (bundled with ZED SDK installer, not installable via pip)
- `numpy`
- `opencv-contrib-python` (required for `cv2.aruco` AprilTag detection)
- `scipy`
- `matplotlib`
- A ZED stereo camera (for live capture and 3D tracking)
- An AprilTag (36h11 family, default size 0.165 m) visible in frame for world-coordinate alignment

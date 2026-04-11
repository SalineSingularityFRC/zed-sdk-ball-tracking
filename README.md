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

## Building the `inekf` submodule

After cloning, initialize submodules and apply the local patch (required if your repo path contains spaces — the upstream `create_symlinks` macro escapes paths in a way that breaks `ln`):

```sh
git submodule update --init --recursive
git -C inekf apply ../patches/inekf-symlink-spaces.patch
mkdir -p inekf/build && cd inekf/build
cmake .. -DPYTHON=ON
make -j
```

If the patch has already been applied, `git apply` will fail; check with `git -C inekf apply --check ../patches/inekf-symlink-spaces.patch` first, or use `git -C inekf apply --reverse --check …` to detect that case.

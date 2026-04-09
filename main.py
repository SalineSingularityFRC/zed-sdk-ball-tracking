import argparse
import sys
from pathlib import Path
import numpy as np
import cv2

from zed_utils import sl, _has_zed
from tracker import BallTracker
from calibration import run_calibration, load_hsv_config


def _show_roi_summary(background, tracker):
    """Show a final image with all ROI trajectories overlaid. Press any key to close."""
    vis = background.copy()
    # Dim the background so trajectories stand out
    vis = (vis * 0.4).astype(np.uint8)

    # Draw ROI
    rx, ry, rw, rh = tracker.roi
    cv2.rectangle(vis, (rx, ry), (rx + rw, ry + rh), (0, 255, 255), 2)

    for s in tracker.roi_stats:
        color = s['color']
        positions = s['positions']
        coeffs_x = s['coeffs_x']
        coeffs_y = s['coeffs_y']
        frames = s['frames']

        # Draw raw detection points
        for px, py in positions:
            cv2.circle(vis, (int(px), int(py)), 4, color, -1, cv2.LINE_AA)

        # Draw fitted curve
        if coeffs_x is not None and coeffs_y is not None and len(frames) >= 3:
            f0, f1 = frames[0], frames[-1]
            curve_pts = []
            for fi in np.linspace(f0, f1, num=max(50, (f1 - f0) * 3)):
                x = np.polyval(coeffs_x, fi)
                y = np.polyval(coeffs_y, fi)
                curve_pts.append((int(x), int(y)))
            if len(curve_pts) > 1:
                cv2.polylines(vis, [np.array(curve_pts, np.int32)], False,
                              color, 2, cv2.LINE_AA)

        # Label at the midpoint of the trajectory
        mid = positions[len(positions) // 2]
        label = f"#{s['id']} {s['airtime']:.2f}s v={s['speed']:.0f}"
        cv2.putText(vis, label, (int(mid[0]) + 8, int(mid[1]) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    # Summary text at top
    cv2.putText(vis, f"ROI Summary: {len(tracker.roi_stats)} trajectories",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    airtimes = [s['airtime'] for s in tracker.roi_stats]
    speeds = [s['speed'] for s in tracker.roi_stats]
    cv2.putText(vis, f"Airtime avg={np.mean(airtimes):.2f}s  Speed avg={np.mean(speeds):.0f}",
                (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow('ROI Summary', vis)
    print("\nShowing ROI summary — press any key to close.")
    cv2.waitKey(0)
    cv2.destroyWindow('ROI Summary')


def run_tracker(video_path=None, camera_index=0, start_from=0.0, no_roi=False):
    lower_hsv, upper_hsv = load_hsv_config()

    kwargs = {}
    if lower_hsv and upper_hsv:
        kwargs['lower_hsv'] = lower_hsv
        kwargs['upper_hsv'] = upper_hsv
    else:
        print("Using default HSV values (run with --calibrate to tune)")

    use_zed = False
    cap = None
    if video_path:
        if not Path(video_path).exists():
            print(f"Error: Video file '{video_path}' not found"); return
        cap = cv2.VideoCapture(video_path)
    else:
        # Prefer ZED camera when available
        if _has_zed:
            try:
                cam = sl.Camera()
                init = sl.InitParameters()
                status = cam.open(init)
                if status == sl.ERROR_CODE.SUCCESS:
                    runtime = sl.RuntimeParameters()
                    cap = cam
                    use_zed = True
                else:
                    cam.close()
                    cap = None
            except Exception:
                cap = None

        if cap is None:
            cap = cv2.VideoCapture(camera_index)

    # Validate capture
    try:
        opened = cap.isOpened() if not (use_zed and isinstance(cap, sl.Camera)) else True
    except Exception:
        opened = False
    if not opened:
        print("Error: Could not open video source"); return

    # Determine fps
    fps = None
    if use_zed and isinstance(cap, sl.Camera):
        try:
            info = cap.get_camera_information()
            fps = getattr(info.camera_configuration, 'fps', None)
        except Exception:
            fps = None
    else:
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
        except Exception:
            fps = None

    if not fps or fps <= 0:
        fps = 30.0
    kwargs['fps'] = fps

    if start_from > 0 and video_path:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_from * 1000.0)
        actual = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        print(f"Seeked to {actual:.1f}s (requested {start_from:.1f}s)")

    # --- ROI selection from first frame ---
    # Read first frame (handle ZED differently)
    if use_zed and isinstance(cap, sl.Camera):
        mat = sl.Mat()
        if cap.grab(runtime) != sl.ERROR_CODE.SUCCESS:
            print("Error: Could not read first frame from ZED"); cap.close(); return
        cap.retrieve_image(mat, sl.VIEW.LEFT)
        first_frame = mat.get_data()
        if first_frame is None:
            print("Error: Could not read first frame from ZED"); cap.close(); return
        if first_frame.ndim == 3 and first_frame.shape[2] == 4:
            first_frame = cv2.cvtColor(first_frame, cv2.COLOR_RGBA2BGR)
    else:
        ret, first_frame = cap.read()
        if not ret:
            print("Error: Could not read first frame");
            if hasattr(cap, 'release'):
                cap.release()
            return

    if not no_roi:
        print("Select ROI and press ENTER/SPACE. Press 'c' to cancel (no ROI).")
        roi = cv2.selectROI("Select ROI", first_frame, showCrosshair=True, fromCenter=False)
        cv2.destroyWindow("Select ROI")
        if roi[2] > 0 and roi[3] > 0:
            kwargs['roi'] = tuple(int(v) for v in roi)
            print(f"ROI set: x={roi[0]} y={roi[1]} w={roi[2]} h={roi[3]}")
        else:
            print("No ROI selected, tracking without ROI")

    tracker = BallTracker(**kwargs)
    # If we opened a ZED camera, enable ZED helpers on the tracker and fetch initial depth
    if use_zed and isinstance(cap, sl.Camera):
        try:
            tracker.enable_zed(cap, runtime)
            # attempt to retrieve an initial depth map
            if tracker._zed_depth_mat is not None:
                try:
                    cap.retrieve_measure(tracker._zed_depth_mat, sl.MEASURE.DEPTH)
                except Exception:
                    pass
        except Exception:
            pass

    # Process the first frame we already read
    mask = tracker.segment(first_frame)
    vis = tracker.track(first_frame, mask)
    cv2.imshow('Ball Tracker', vis)
    cv2.imshow('Mask', mask)

    print("Press 'q' to quit")
    try:
        while True:
            # Read frame depending on capture type
            if use_zed and isinstance(cap, sl.Camera):
                mat = sl.Mat()
                if cap.grab(runtime) != sl.ERROR_CODE.SUCCESS:
                    print("End of ZED stream"); break
                cap.retrieve_image(mat, sl.VIEW.LEFT)
                frame = mat.get_data()
                if frame is None:
                    print("End of ZED stream"); break
                if frame.ndim == 3 and frame.shape[2] == 4:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                # update depth map for this frame so tracker can query 3D points
                try:
                    if tracker._zed_depth_mat is not None:
                        cap.retrieve_measure(tracker._zed_depth_mat, sl.MEASURE.DEPTH)
                except Exception:
                    # non-fatal; continue without depth
                    pass
            else:
                ret, frame = cap.read()
                if not ret:
                    print("End of video"); break

            mask = tracker.segment(frame)
            vis = tracker.track(frame, mask)

            cv2.imshow('Ball Tracker', vis)
            cv2.imshow('Mask', mask)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # Cleanup capture
        if cap is not None:
            try:
                if use_zed and isinstance(cap, sl.Camera):
                    cap.close()
                elif hasattr(cap, 'release'):
                    cap.release()
            except Exception:
                pass
        cv2.destroyAllWindows()

        # Flush any still-active tracks into ROI stats
        if tracker.roi is not None:
            for t in tracker.tracks:
                tracker._capture_roi_stats(t)

        # Print ROI summary and show final trajectory visualization
        if tracker.roi_stats:
            print(f"\n{'='*50}")
            print(f"ROI Summary: {len(tracker.roi_stats)} balls detected")
            print(f"{'='*50}")
            print(f"{'ID':>4}  {'Airtime':>8}  {'Speed':>6}  {'Disp':>7}  {'Pts':>4}  {'Err':>5}")
            for s in tracker.roi_stats:
                print(f"#{s['id']:>3}  {s['airtime']:>7.2f}s  {s['speed']:>6.0f}  "
                      f"{s['displacement']:>6.0f}px  {s['detections']:>4}  {s['fit_error']:>5.1f}")
            airtimes = [s['airtime'] for s in tracker.roi_stats]
            speeds = [s['speed'] for s in tracker.roi_stats]
            print(f"\nAirtime  avg={np.mean(airtimes):.2f}s  "
                  f"min={min(airtimes):.2f}s  max={max(airtimes):.2f}s")
            print(f"Speed    avg={np.mean(speeds):.0f}  "
                  f"min={min(speeds):.0f}  max={max(speeds):.0f}")

            _show_roi_summary(first_frame, tracker)


def main():
    parser = argparse.ArgumentParser(description='Multi-ball ballistic tracker')
    parser.add_argument('--calibrate', action='store_true')
    parser.add_argument('--video', type=str)
    parser.add_argument('--image', type=str)
    parser.add_argument('--camera', type=int, default=1)
    parser.add_argument('--start-from', type=float, default=0.0,
                        help='Skip to this many seconds into the video')
    parser.add_argument('--no-roi', action='store_true',
                        help='Skip ROI selection')
    args = parser.parse_args()

    if args.calibrate:
        run_calibration(video_path=args.video, image_path=args.image, camera_index=args.camera)
    else:
        run_tracker(video_path=args.video, camera_index=args.camera,
                    start_from=args.start_from, no_roi=args.no_roi)


if __name__ == '__main__':
    sys.exit(main())

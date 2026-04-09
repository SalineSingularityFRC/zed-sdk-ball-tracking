import argparse
import sys
from pathlib import Path
import numpy as np
import cv2

try:
    import matplotlib.pyplot as plt
    _has_plt = True
except ImportError:
    _has_plt = False

from zed_utils import sl, _has_zed
from tracker import BallTracker
from calibration import run_calibration, load_hsv_config


def _show_roi_summary(first_frame, tracker):
    """
    Displays the final summary with tracks, and optionally plots 3D trajectories
    if a ZED camera and depth data were used.
    """
    h_max = first_frame.shape[0] if first_frame is not None else 480
    w_max = first_frame.shape[1] if first_frame is not None else 640
    
    summary = np.zeros((h_max, w_max, 3), dtype=np.uint8)
    has_3d = False
    
    for s in tracker.roi_stats:
        b, g, r = s['color']
        c = (int(b), int(g), int(r))
        pts = s['positions']
        for p in pts:
            cv2.circle(summary, (int(p[0]), int(p[1])), 4, c, -1)
            
        pos3d = s.get('positions_3d')
        if pos3d and any(p is not None for p in pos3d):
            has_3d = True
            
        # Optional: Print track info
        #print(f"Track ID #{s['id']} - Airtime: {s['airtime']:.2f}s, Speed: {s['speed']:.1f}px/frame")
    
    cv2.putText(summary, f"ROI Summary: {len(tracker.roi_stats)} trajectories",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    airtimes = [s['airtime'] for s in tracker.roi_stats]
    speeds = [s['speed'] for s in tracker.roi_stats]
    cv2.putText(summary, f"Airtime avg={np.mean(airtimes):.2f}s  Speed avg={np.mean(speeds):.0f}",
                (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow('ROI Summary', summary)
    
    _has_plt = False
    try:
        import matplotlib.pyplot as plt
        _has_plt = True
    except ImportError:
        pass
    
    plotted_any = False
    
    if _has_plt:
        if has_3d:
            print("\nOpening 3D Matplotlib trajectory summary. Close the plot window to finish.")
        else:
            print("\nNo ZED depth data found. Showing semi-3D plot (X, Time, Y) instead.")
            
        plt.ion() # Enable interactive mode
        fig = plt.figure("3D Trajectories")
        ax = fig.add_subplot(111, projection='3d')
        
        for s in tracker.roi_stats:
            pos3d = s.get('positions_3d')
            pos2d = s.get('positions')
            frames = s.get('frames')
            if not pos2d: continue
            
            b, g, r = s['color']
            color_hex = '#%02x%02x%02x' % (r, g, b)
            
            X, Y, Z = [], [], []
            if has_3d and pos3d:
                valid = [(p[0], p[1], p[2]) for p in pos3d if p is not None]
                if valid:
                    X = [p[0] for p in valid]
                    Y = [p[1] for p in valid]
                    Z = [p[2] for p in valid]
            
            if not X: # Fallback to 2D (X, Time, Y) if no 3D points in this track
                X = [p[0] for p in pos2d]
                Z = frames  # Use frame time as Z depth
                Y = [p[1] for p in pos2d]

            if not X: continue

            ax.scatter(X, Z, Y, color=color_hex, label=f"ID #{s['id']}")
            plotted_any = True
            
            # Plot polynomial curve
            cX, cY, cZ = s.get('coeffs_X'), s.get('coeffs_Y'), s.get('coeffs_Z')
            cx, cy = s.get('coeffs_x'), s.get('coeffs_y')
            
            if has_3d and cX is not None and cY is not None and cZ is not None and len(frames) >= 2:
                t_vals = np.linspace(frames[0], frames[-1], 50)
                curve_X = np.polyval(cX, t_vals)
                curve_Y = np.polyval(cY, t_vals)
                curve_Z = np.polyval(cZ, t_vals)
                ax.plot(curve_X, curve_Z, curve_Y, color=color_hex)
            elif not has_3d and cx is not None and cy is not None and len(frames) >= 2:
                # 2D fallback poly
                t_vals = np.linspace(frames[0], frames[-1], 50)
                curve_X = np.polyval(cx, t_vals)
                curve_Y = np.polyval(cy, t_vals)
                curve_Z = t_vals
                ax.plot(curve_X, curve_Z, curve_Y, color=color_hex)
                
        if plotted_any:
            if has_3d:
                ax.set_xlabel('X (m) [Right]')
                ax.set_ylabel('Z (m) [Forward]')
                ax.set_zlabel('Y (m) [Down]')
            else:
                ax.set_xlabel('X (px)')
                ax.set_ylabel('Frame (Time)')
                ax.set_zlabel('Y (px)')
                
            ax.invert_zaxis() # Invert visually so Y drops down
            
            try:
                ax.set_box_aspect((1, 1, 1))
            except AttributeError:
                pass

            plt.legend()
            plt.show(block=False)  # Show without blocking so OpenCV remains responsive

    if not _has_plt:
        print("\n'matplotlib' is missing! Run 'pip install matplotlib' to see graph visualizations.")
        
    print("\nShowing ROI summary — press any key in the OpenCV window to close.")
    
    if _has_plt and plotted_any:
        # Event loop to keep both Matplotlib and OpenCV windows responsive simultaneously
        while True:
            # OpenCV waitKey waits briefly for a key press
            key = cv2.waitKey(50)
            if key != -1:
                break
            # Process Matplotlib events so the window stays active
            try:
                if plt.fignum_exists(fig.number):
                    fig.canvas.flush_events()
                else:
                    # if they close the matplotlib window, exit
                    break
            except Exception:
                pass
    else:
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

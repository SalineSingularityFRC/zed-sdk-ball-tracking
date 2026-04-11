import argparse
import sys
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt

from zed_utils import sl
from tracker import BallTracker
from calibration import run_calibration, load_hsv_config
from localization.camera_localizer import CameraLocalizer
from localization.ekf import load_tag_world_poses
from visualization import (show_final_tracks, compute_target_center,
                           target_plane_z, compute_track_miss)
from config import Config
from nt.nt_client import NTRobotClient
from nt.nt_recorder import NTRecorder
from nt.nt_log import NTLog
from analysis.track_world import world_points_for_stat
from analysis.shot_logger import ShotLogger
from analysis import miss_features as mf

TAG_SIZE_M = 0.1651
TAGS_JSON_PATH = Path("localization/tags.json")


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

    plotted_any = False

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
        ax.set_box_aspect((1, 1, 1))

        plt.legend()
        plt.show(block=False)  # Show without blocking so OpenCV remains responsive

    print("\nShowing ROI summary — press any key in the OpenCV window to close.")

    if plotted_any:
        # Event loop to keep both Matplotlib and OpenCV windows responsive simultaneously
        while True:
            # OpenCV waitKey waits briefly for a key press
            key = cv2.waitKey(50)
            if key != -1:
                break
            # Process Matplotlib events so the window stays active
            if plt.fignum_exists(fig.number):
                fig.canvas.flush_events()
            else:
                # if they close the matplotlib window, exit
                break
    else:
        cv2.waitKey(0)

    cv2.destroyWindow('ROI Summary')


def _make_zed_init(svo_path=None):
    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.HD720
    init.camera_fps = 60
    init.depth_mode = sl.DEPTH_MODE.NEURAL
    init.coordinate_units = sl.UNIT.MILLIMETER
    if svo_path is not None:
        init.set_from_svo_file(str(svo_path))
        init.svo_real_time_mode = False
    return init


def record_svo(output_path):
    """Step 1: Open the ZED and record an SVO file. No processing."""
    out = Path(output_path)
    if out.parent and str(out.parent) not in ('', '.'):
        out.parent.mkdir(parents=True, exist_ok=True)

    cam = sl.Camera()
    init = _make_zed_init()
    status = cam.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"Error: Could not open ZED camera: {status}")
        cam.close()
        return

    rec_params = sl.RecordingParameters(str(out), sl.SVO_COMPRESSION_MODE.H264)
    err = cam.enable_recording(rec_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"Error: enable_recording failed: {err}")
        cam.close()
        return

    cfg = Config.load()
    nt_client = None
    nt_recorder = None
    if cfg.nt.keys:
        try:
            nt_client = NTRobotClient(cfg.nt)
            nt_recorder = NTRecorder(nt_client, str(out) + ".nt.jsonl")
        except Exception as e:
            print(f"[nt] recorder disabled: {e}")
            nt_client = None
            nt_recorder = None
    else:
        print("[nt] no keys configured; skipping NT sidecar")

    runtime = sl.RuntimeParameters()
    frame_count = 0
    print(f"Recording HD720@60 NEURAL to {out}. Press Ctrl+C to stop.")
    try:
        while True:
            if cam.grab(runtime) != sl.ERROR_CODE.SUCCESS:
                continue
            if nt_recorder is not None:
                try:
                    nt_recorder.record_frame(frame_count)
                except Exception as e:
                    print(f"[nt] record_frame failed: {e}")
            frame_count += 1
            if frame_count % 60 == 0:
                print(f"  recorded {frame_count} frames", end='\r', flush=True)
    except KeyboardInterrupt:
        print()
    finally:
        if nt_recorder is not None:
            nt_recorder.close()
        if nt_client is not None:
            nt_client.close()
        cam.disable_recording()
        cam.close()
        print(f"Recorded {frame_count} frames to {out}")


def _grab_single_frame(svo_path=None, video_path=None, image_path=None,
                       camera_index=0, frame_index=0):
    """Return a single BGR frame from whichever source is provided."""
    if image_path:
        if not Path(image_path).exists():
            print(f"Error: Image file '{image_path}' not found"); return None
        return cv2.imread(image_path)
    if svo_path:
        if not Path(svo_path).exists():
            print(f"Error: SVO file '{svo_path}' not found"); return None
        cam = sl.Camera()
        init = _make_zed_init(svo_path=svo_path)
        if cam.open(init) != sl.ERROR_CODE.SUCCESS:
            print("Error: Could not open SVO"); return None
        total = cam.get_svo_number_of_frames()
        target = max(0, min(frame_index, total - 1)) if total > 0 else max(0, frame_index)
        if target > 0:
            cam.set_svo_position(target)
        runtime = sl.RuntimeParameters()
        mat = sl.Mat()
        if cam.grab(runtime) != sl.ERROR_CODE.SUCCESS:
            cam.close(); print(f"Error: could not grab frame {target}"); return None
        cam.retrieve_image(mat, sl.VIEW.LEFT)
        frame = mat.get_data()
        cam.close()
        if frame is None:
            return None
        if frame.ndim == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
        print(f"Grabbed SVO frame {target} of {total}")
        return frame
    if video_path:
        if not Path(video_path).exists():
            print(f"Error: Video file '{video_path}' not found"); return None
        cap = cv2.VideoCapture(video_path)
        if frame_index > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_index))
        ret, frame = cap.read()
        cap.release()
        return frame if ret else None
    # live camera fallback
    cam = sl.Camera()
    init = _make_zed_init()
    if cam.open(init) == sl.ERROR_CODE.SUCCESS:
        runtime = sl.RuntimeParameters()
        mat = sl.Mat()
        frame = None
        if cam.grab(runtime) == sl.ERROR_CODE.SUCCESS:
            cam.retrieve_image(mat, sl.VIEW.LEFT)
            frame = mat.get_data()
            if frame is not None and frame.ndim == 3 and frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
        cam.close()
        return frame
    cap = cv2.VideoCapture(camera_index)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def inspect_contours(svo_path=None, video_path=None, image_path=None,
                     camera_index=0, frame_index=0):
    """Grab one frame, segment it, and label every contour with its circularity."""
    frame = _grab_single_frame(svo_path=svo_path, video_path=video_path,
                               image_path=image_path, camera_index=camera_index,
                               frame_index=frame_index)
    if frame is None:
        print("Error: no frame captured"); return

    lower_hsv, upper_hsv = load_hsv_config()
    kwargs = {}
    if lower_hsv and upper_hsv:
        kwargs['lower_hsv'] = lower_hsv
        kwargs['upper_hsv'] = upper_hsv
    tracker = BallTracker(**kwargs)
    mask = tracker.segment(frame)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    vis = frame.copy()
    min_area = np.pi * (tracker.min_radius ** 2) * 0.8
    print(f"\nFound {len(contours)} contours "
          f"(min_area={min_area:.0f}, min_circ={tracker.min_circularity}, "
          f"min_r={tracker.min_radius}, max_r={tracker.max_radius})")
    print(f"{'idx':>4} {'area':>8} {'perim':>8} {'circ':>6} {'radius':>7} {'axis':>6}"
          f" {'fill':>6}  verdict")

    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)
        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0.0
        (cx, cy), radius = cv2.minEnclosingCircle(c)
        cx_i, cy_i = int(cx), int(cy)

        # Replicate tracker._detect_circles checks in order, collect failures
        reasons = []
        if area < min_area:
            reasons.append(f"area<{min_area:.0f}")
        if perimeter == 0:
            reasons.append("perim=0")
        if perimeter > 0 and circularity < tracker.min_circularity:
            reasons.append(f"circ<{tracker.min_circularity}")

        axis_ratio = None
        if len(c) >= 5:
            (_, (minor, major), _) = cv2.fitEllipse(c)
            axis_ratio = minor / (major + 1e-6)
            if axis_ratio < 0.65:
                reasons.append(f"axis<0.65({axis_ratio:.2f})")

        circle_area = np.pi * radius * radius
        fill = area / (circle_area + 1e-6)
        if fill < 0.60:
            reasons.append(f"fill<0.60({fill:.2f})")

        if radius < tracker.min_radius:
            reasons.append(f"r<{tracker.min_radius}")
        elif radius > tracker.max_radius:
            reasons.append(f"r>{tracker.max_radius}")

        passes = not reasons
        color = (0, 255, 0) if passes else (0, 0, 255)

        cv2.drawContours(vis, [c], -1, color, 2)
        cv2.circle(vis, (cx_i, cy_i), 3, color, -1)
        cv2.putText(vis, f"#{i} c{circularity:.2f}", (cx_i + 6, cy_i - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        cv2.putText(vis, f"r{radius:.0f}", (cx_i + 6, cy_i + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

        axis_str = f"{axis_ratio:.2f}" if axis_ratio is not None else "  -- "
        verdict = "OK" if passes else ",".join(reasons)
        print(f"{i:>4} {area:>8.0f} {perimeter:>8.1f} {circularity:>6.2f} {radius:>7.1f}"
              f" {axis_str:>6} {fill:>6.2f}  {verdict}")

    cv2.imshow('Contour Inspect', vis)
    cv2.imshow('Mask', mask)
    print("\nGreen = passes filters, red = rejected. Press any key to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def _log_shots(svo_path, stat_to_world, tag_world_poses):
    """For each track that crosses the target plane downward, look up the
    NetworkTables sidecar at the track's start frame, build features, and
    append a row to shots.csv. Silently no-ops if no sidecar exists."""
    if not stat_to_world or svo_path is None:
        return

    log = NTLog.load(str(svo_path) + ".nt.jsonl")
    if log is None or not log.has_data:
        print("[shots] no NT sidecar found; skipping shot logging")
        return

    cfg = Config.load()

    if cfg.target.has_override:
        target = (cfg.target.x, cfg.target.y, cfg.target.z)
    else:
        target = compute_target_center(tag_world_poses)
        if target is None:
            print("[shots] no target center available; skipping shot logging")
            return
    target_x, target_y, target_z = target

    plane_z = target_z + 0.2  # match TARGET_PLANE_OFFSET_M in visualization.py
    target_xy = (target_x, target_y)

    logger = ShotLogger(cfg.logging.shots_csv)
    n_logged = 0

    for stat, pts in stat_to_world:
        crossing = compute_track_miss(pts, target_xy, plane_z)
        if crossing is None:
            continue
        miss_x, miss_y, _ = crossing

        frame_start = int(stat['frames'][0])
        frame_cross = int(stat['frames'][-1])
        nt_row = log.at(frame_start)
        if nt_row is None:
            continue

        def gv(key, default=0.0):
            v = nt_row.get(key)
            return float(v) if v is not None else default

        robot_x = gv('pose_x')
        robot_y = gv('pose_y')
        robot_theta = gv('pose_theta')
        vel_x = gv('vel_x')
        vel_y = gv('vel_y')
        omega = gv('omega')
        turret_angle = gv('turret_angle')

        turret = mf.turret_robot_frame(
            robot_x, robot_y, 0.0, robot_theta,
            cfg.turret.offset_x, cfg.turret.offset_y, cfg.turret.offset_z,
            turret_angle,
        )
        rng, bearing = mf.turret_relative_polar(turret, target_x, target_y)
        v_rad, v_tan = mf.decompose_velocity(vel_x, vel_y, turret, target_x, target_y)
        miss_r, miss_theta = mf.miss_in_polar(miss_x, miss_y, turret, target_x, target_y)

        logger.append({
            'shot_id': stat['id'],
            'svo': str(svo_path),
            'frame_start': frame_start,
            'frame_cross': frame_cross,
            'robot_x': robot_x,
            'robot_y': robot_y,
            'robot_theta': robot_theta,
            'vel_x': vel_x,
            'vel_y': vel_y,
            'omega': omega,
            'turret_angle': turret_angle,
            'turret_world_x': turret.x,
            'turret_world_y': turret.y,
            'turret_world_theta': turret.theta,
            'target_x': target_x,
            'target_y': target_y,
            'target_z': target_z,
            'range_to_target': rng,
            'bearing_to_target': bearing,
            'v_radial': v_rad,
            'v_tangential': v_tan,
            'miss_x': miss_x,
            'miss_y': miss_y,
            'miss_r': miss_r,
            'miss_theta': miss_theta,
        })
        n_logged += 1

    print(f"[shots] logged {n_logged} shots to {cfg.logging.shots_csv}")


def run_tracker(video_path=None, svo_path=None, camera_index=0, start_from=0.0, no_roi=False):
    lower_hsv, upper_hsv = load_hsv_config()

    kwargs = {}
    if lower_hsv and upper_hsv:
        kwargs['lower_hsv'] = lower_hsv
        kwargs['upper_hsv'] = upper_hsv
    else:
        print("Using default HSV values (run with --calibrate to tune)")

    use_zed = False
    cap = None
    runtime = None
    if svo_path:
        if not Path(svo_path).exists():
            print(f"Error: SVO file '{svo_path}' not found"); return
        cam = sl.Camera()
        init = _make_zed_init(svo_path=svo_path)
        status = cam.open(init)
        if status != sl.ERROR_CODE.SUCCESS:
            print(f"Error: Could not open SVO file: {status}"); return
        runtime = sl.RuntimeParameters()
        cap = cam
        use_zed = True
    elif video_path:
        if not Path(video_path).exists():
            print(f"Error: Video file '{video_path}' not found"); return
        cap = cv2.VideoCapture(video_path)
    else:
        # Prefer ZED camera when available
        cam = sl.Camera()
        init = _make_zed_init()
        status = cam.open(init)
        if status == sl.ERROR_CODE.SUCCESS:
            runtime = sl.RuntimeParameters()
            cap = cam
            use_zed = True
        else:
            cam.close()
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
            first_frame = cv2.cvtColor(first_frame, cv2.COLOR_RGBA2RGB)
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
            if getattr(tracker, '_zed_pc_mat', None) is not None:
                try:
                    cap.retrieve_measure(tracker._zed_pc_mat, sl.MEASURE.XYZRGBA)
                except Exception:
                    pass
        except Exception:
            pass

    # Localizer runs alongside the tracker, sharing the same ZED camera.
    localizer = None
    tag_world_poses = {}
    if use_zed and isinstance(cap, sl.Camera) and TAGS_JSON_PATH.exists():
        try:
            localizer = CameraLocalizer(zed=cap, tag_size=TAG_SIZE_M,
                                        tags_path=str(TAGS_JSON_PATH))
            tag_world_poses = load_tag_world_poses(TAGS_JSON_PATH)
            print(f"[localizer] enabled with {len(tag_world_poses)} tags from {TAGS_JSON_PATH}")
        except Exception as e:
            print(f"[localizer] disabled: {e}")
            localizer = None

    # Process the first frame we already read
    if localizer is not None:
        try:
            localizer.step_external(first_frame)
        except Exception as e:
            print(f"[localizer] step failed: {e}")

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
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
                # update point cloud for this frame so tracker can query 3D points
                try:
                    if getattr(tracker, '_zed_pc_mat', None) is not None:
                        cap.retrieve_measure(tracker._zed_pc_mat, sl.MEASURE.XYZRGBA)
                except Exception:
                    # non-fatal; continue without depth
                    pass
            else:
                ret, frame = cap.read()
                if not ret:
                    print("End of video"); break

            if localizer is not None:
                try:
                    localizer.step_external(frame)
                except Exception:
                    pass

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
            print(f"\n[flush] {len(tracker.tracks)} tracks still active at end of stream")
            for t in tracker.tracks:
                traj = t.trajectory
                len_ok = traj.length >= tracker.min_track_length
                disp_ok = t.total_displacement >= tracker.min_displacement
                fit_ok = traj.fit_error <= tracker.max_fit_error
                hit = tracker._track_hits_roi(t)
                print(f"  track #{t.id}: len={traj.length} disp={t.total_displacement:.0f} "
                      f"fit_err={traj.fit_error:.1f} hits_roi={hit} "
                      f"valid={len_ok and disp_ok and fit_ok}")
                tracker._capture_roi_stats(t)
            print(f"[flush] captured {len(tracker.roi_stats)} roi_stats total")

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

            if localizer is not None:
                world_pose = localizer.pose()
                tracks_world = []
                stat_to_world: list[tuple[dict, list[tuple[float, float, float]]]] = []
                for s in tracker.roi_stats:
                    pts_world = world_points_for_stat(s, world_pose)
                    if pts_world is None:
                        continue
                    b, g, r = s['color']
                    tracks_world.append({
                        'points': pts_world,
                        'color': (r / 255.0, g / 255.0, b / 255.0),
                    })
                    stat_to_world.append((s, pts_world))

                _log_shots(svo_path, stat_to_world, tag_world_poses)

                if tracks_world:
                    print(f"\nOpening VTK viewer with {len(tracks_world)} tracks in field frame…")
                    show_final_tracks(tag_world_poses, TAG_SIZE_M, tracks_world)
                else:
                    print("\nNo 3D track points available; falling back to 2D summary.")
                    _show_roi_summary(first_frame, tracker)
            else:
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
    parser.add_argument('--record', type=str, metavar='PATH.svo2',
                        help='Step 1: record a ZED SVO file at HD720@60 NEURAL and exit')
    parser.add_argument('--svo', type=str, metavar='PATH.svo2',
                        help='Step 2: process a previously recorded SVO file')
    parser.add_argument('--calib-frame', type=int, default=0,
                        help='Frame index to calibrate on (for --svo/--video calibration)')
    parser.add_argument('--inspect-contours', action='store_true',
                        help='Grab a single frame and label every contour with its circularity')
    args = parser.parse_args()

    if args.record:
        record_svo(args.record)
    elif args.inspect_contours:
        inspect_contours(svo_path=args.svo, video_path=args.video, image_path=args.image,
                         camera_index=args.camera, frame_index=args.calib_frame)
    elif args.calibrate:
        run_calibration(video_path=args.video, image_path=args.image,
                        svo_path=args.svo, camera_index=args.camera,
                        calib_frame=args.calib_frame)
    else:
        run_tracker(video_path=args.video, svo_path=args.svo, camera_index=args.camera,
                    start_from=args.start_from, no_roi=args.no_roi)


if __name__ == '__main__':
    sys.exit(main())

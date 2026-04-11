import cv2
import numpy as np
from pathlib import Path
from zed_utils import sl

def load_hsv_config():
    config_path = Path('hsv_config.txt')
    if not config_path.exists():
        return None, None
    try:
        lower, upper = None, None
        for line in config_path.read_text().splitlines():
            line = line.strip()
            if line.startswith('lower_hsv'):
                lower = [int(x.strip()) for x in line.split('=')[1].strip().strip('[]').split(',')]
            elif line.startswith('upper_hsv'):
                upper = [int(x.strip()) for x in line.split('=')[1].strip().strip('[]').split(',')]
        if lower and upper:
            print(f"Loaded HSV config: lower={lower} upper={upper}")
            return lower, upper
    except Exception as e:
        print(f"Warning: could not load HSV config: {e}")
    return None, None

def run_calibration(video_path=None, image_path=None, svo_path=None,
                    camera_index=0, calib_frame=0):
    def nothing(x):
        pass

    if image_path:
        if not Path(image_path).exists():
            print(f"Error: Image file '{image_path}' not found"); return
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Error: Could not load image '{image_path}'"); return
        cap = None
    elif svo_path:
        if not Path(svo_path).exists():
            print(f"Error: SVO file '{svo_path}' not found"); return
        cam = sl.Camera()
        init = sl.InitParameters()
        init.set_from_svo_file(str(svo_path))
        init.svo_real_time_mode = False
        init.camera_resolution = sl.RESOLUTION.HD720
        init.camera_fps = 60
        init.depth_mode = sl.DEPTH_MODE.NEURAL
        init.coordinate_units = sl.UNIT.MILLIMETER
        status = cam.open(init)
        if status != sl.ERROR_CODE.SUCCESS:
            print(f"Error: Could not open SVO file: {status}"); return
        total = cam.get_svo_number_of_frames()
        target = max(0, min(calib_frame, total - 1)) if total > 0 else max(0, calib_frame)
        if target > 0:
            cam.set_svo_position(target)
        runtime = sl.RuntimeParameters()
        mat = sl.Mat()
        if cam.grab(runtime) != sl.ERROR_CODE.SUCCESS:
            print(f"Error: Could not grab frame {target} from SVO"); cam.close(); return
        cam.retrieve_image(mat, sl.VIEW.LEFT)
        frame = mat.get_data()
        cam.close()
        if frame is None:
            print("Error: retrieved empty frame from SVO"); return
        if frame.ndim == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
        print(f"Calibrating on SVO frame {target} (of {total})")
        cap = None
    elif video_path:
        if not Path(video_path).exists():
            print(f"Error: Video file '{video_path}' not found"); return
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video '{video_path}'"); return
        if calib_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(calib_frame))
            ret, frame = cap.read()
            cap.release()
            cap = None
            if not ret or frame is None:
                print(f"Error: Could not read frame {calib_frame} from video"); return
            print(f"Calibrating on video frame {calib_frame}")
    else:
        # Prefer ZED camera when available
        use_zed = False
        cap = None
        cam = sl.Camera()
        init = sl.InitParameters()
        # Use default init params; user can tune if needed
        status = cam.open(init)
        if status == sl.ERROR_CODE.SUCCESS:
            runtime = sl.RuntimeParameters()
            mat = sl.Mat()
            # small warm-up and single frame grab
            if cam.grab(runtime) == sl.ERROR_CODE.SUCCESS:
                cam.retrieve_image(mat, sl.VIEW.LEFT)
                frame = mat.get_data()
                # Convert RGBA (ZED) to BGR for OpenCV if needed
                if frame is not None and frame.ndim == 3 and frame.shape[2] == 4:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
                use_zed = True
                cap = cam
            else:
                cam.close()
        else:
            cam.close()

        if cap is None:
            cap = cv2.VideoCapture(camera_index)
            if not cap.isOpened():
                print(f"Error: Could not open camera {camera_index}"); return
            import time; time.sleep(0.5)
            ret, frame = cap.read()
            if not ret:
                print("Error: Camera opened but cannot read frames");
                # cleanup
                if hasattr(cap, 'release'):
                    cap.release()
                return

    window_name = 'HSV Calibration Tool'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    for name, mx, default in [('H Min', 179, 0), ('H Max', 179, 179),
                               ('S Min', 255, 0), ('S Max', 255, 255),
                               ('V Min', 255, 0), ('V Max', 255, 255)]:
        cv2.createTrackbar(name, window_name, default, mx, nothing)

    print("\nHSV Calibration — 'q' quit, 's' save, 'r' reset")

    try:
        while True:
            if cap is not None:
                # If using ZED camera instance
                if isinstance(cap, sl.Camera):
                    if cap.grab(runtime) != sl.ERROR_CODE.SUCCESS:
                        break
                    mat = sl.Mat()
                    cap.retrieve_image(mat, sl.VIEW.LEFT)
                    new_frame = mat.get_data()
                    if new_frame is None:
                        break
                    if new_frame.ndim == 3 and new_frame.shape[2] == 4:
                        new_frame = cv2.cvtColor(new_frame, cv2.COLOR_RGBA2RGB)
                    frame = new_frame
                else:
                    ret, new_frame = cap.read()
                    if not ret:
                        if video_path:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0); continue
                        else:
                            break
                    frame = new_frame

            vals = {n: cv2.getTrackbarPos(n, window_name)
                    for n in ['H Min', 'H Max', 'S Min', 'S Max', 'V Min', 'V Max']}
            lower = np.array([vals['H Min'], vals['S Min'], vals['V Min']])
            upper = np.array([vals['H Max'], vals['S Max'], vals['V Max']])

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower, upper)
            result = cv2.bitwise_and(frame, frame, mask=mask)

            dh = 400
            h, w = frame.shape[:2]
            dw = int(w * dh / h)
            row = np.hstack([cv2.resize(frame, (dw, dh)),
                             cv2.resize(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), (dw, dh)),
                             cv2.resize(result, (dw, dh))])
            cv2.imshow(window_name, row)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print(f"\nLower HSV: [{vals['H Min']}, {vals['S Min']}, {vals['V Min']}]")
                print(f"Upper HSV: [{vals['H Max']}, {vals['S Max']}, {vals['V Max']}]")
                break
            elif key == ord('s'):
                with open('hsv_config.txt', 'w') as f:
                    f.write(f"lower_hsv = [{vals['H Min']}, {vals['S Min']}, {vals['V Min']}]\n")
                    f.write(f"upper_hsv = [{vals['H Max']}, {vals['S Max']}, {vals['V Max']}]\n")
                print("Saved to hsv_config.txt")
            elif key == ord('r'):
                for n, v in [('H Min', 0), ('H Max', 179), ('S Min', 0),
                             ('S Max', 255), ('V Min', 0), ('V Max', 255)]:
                    cv2.setTrackbarPos(n, window_name, v)
    finally:
        # Cleanup capture
        if cap is not None:
            try:
                if isinstance(cap, sl.Camera):
                    cap.close()
                elif hasattr(cap, 'release'):
                    cap.release()
            except Exception:
                pass
        cv2.destroyAllWindows()

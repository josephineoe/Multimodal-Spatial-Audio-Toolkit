# ============================================================================
# VISION MODULE
# Real-time object detection and audio source control
# ============================================================================

import threading
import time
import os
import math

import numpy as np


# =========================================================
# Vision Configuration
# =========================================================

VISION_CONFIG = {
    # Camera source: "default" uses index 0. On Linux/Jetson you can use "/dev/video0".
    "camera_source": "default",   # "default" | "usb" | "gstreamer"
    "gst_pipeline": "",

    # Best-effort camera settings
    "width": 1280,
    "height": 720,
    "fps": 30,
    "use_mjpeg": True,

    # YOLO
    "model_path": "yolo11n.pt",
    "conf_thres": 0.25,
    "infer_hz": 8.0,

    # Target selection mode for vision:
    #  - "allowed_objects": largest box among allowed_classes (includes person)
    #  - "person_only":     largest person box only
    "target_mode": "allowed_objects",

    # Allow-list (includes "person")
    "allowed_classes": {"cup", "chair", "couch", "bed", "dining table", "book", "microwave", "person"},

    # Phase 2 camera model (azimuth-only)
    "hfov_deg": 70.0,

    # Gate / timeout
    "gate_conf_thres": 0.25,
    "no_detection_fade_s": 0.75,

    # Smoothing to reduce jitter (EMA)
    "smooth_beta_az": 0.20,
    "smooth_beta_el": 0.20,
    "smooth_beta_dist": 0.25,

    # Gain shaping from distance
    "distance_ref_m": 1.4,
    "gain_min": 0.3,
    "gain_max": 1.0,
    "gain_smooth_beta": 0.20,

    # Distance estimation
    "distance_mode": "bbox_height",
    "distance_fixed_m": 1.4,
    "distance_min_m": 0.3,
    "distance_max_m": 6.0,
    "distance_smoothing_alpha": 0.25,
    "class_real_heights_m": {
        "person": 1.7,
        "chair": 1.0,
        "couch": 1.0,
        "bed": 0.6,
        "dining table": 0.75,
        "book": 0.25,
        "microwave": 0.35,
        "cup": 0.12,
    },

    # Distance effects toggle
    "enable_distance_attenuation": True,

    # Debug / UI
    "show_window": False,
    "window_name": "Vision (YOLO Phase-3)",

    # Print throttling
    "print_every_n_frames": 30,
}


# =========================================================
# Vision Helper Functions
# =========================================================

def _open_camera_for_vision():
    """Open camera source based on configuration."""
    import cv2

    src = VISION_CONFIG["camera_source"]
    if src == "default":
        cap = cv2.VideoCapture(0)
    elif src == "usb":
        cap = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)
    elif src == "gstreamer":
        gst = VISION_CONFIG["gst_pipeline"]
        if not gst:
            raise ValueError("VISION_CONFIG['gst_pipeline'] is empty.")
        cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
    else:
        raise ValueError(f"Unknown VISION_CONFIG['camera_source']: {src}")

    return cap


def _freeze_camera_settings(cap):
    """Apply camera settings from configuration."""
    import cv2

    if VISION_CONFIG["use_mjpeg"]:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(VISION_CONFIG["width"]))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(VISION_CONFIG["height"]))
    cap.set(cv2.CAP_PROP_FPS, float(VISION_CONFIG["fps"]))

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_str = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])

    print(f"[VISION][CAM] Requested: {VISION_CONFIG['width']}x{VISION_CONFIG['height']}@{VISION_CONFIG['fps']} "
          f"{'MJPG' if VISION_CONFIG['use_mjpeg'] else ''}")
    print(f"[VISION][CAM] Actual:    {actual_w}x{actual_h}@{actual_fps:.2f} FOURCC={fourcc_str}")


def _pixels_to_azimuth_deg(cx, W, hfov_deg):
    """Convert pixel x-coordinate to azimuth angle."""
    nx = (cx - (W / 2.0)) / (W / 2.0)
    return nx * (hfov_deg / 2.0)


def _pick_target_index_xyxy(xyxy, cls_ids, names, mode, allowed_classes):
    """
    Select best target from detections based on area and class.
    xyxy: (N,4) numpy array
    cls_ids: (N,) numpy array int
    names: dict {id: name}
    Returns best index or None.
    """
    if xyxy is None or len(xyxy) == 0:
        return None

    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])

    if mode == "person_only":
        allowed = {"person"}
    elif mode == "allowed_objects":
        allowed = allowed_classes
    else:
        raise ValueError(f"Unknown vision target_mode: {mode}")

    best_i = None
    best_area = -1.0
    for i in range(len(xyxy)):
        cls_name = names.get(int(cls_ids[i]), str(int(cls_ids[i])))
        if cls_name not in allowed:
            continue
        if areas[i] > best_area:
            best_area = areas[i]
            best_i = i
    return best_i


# =========================================================
# ObjectDetectionYOLO: Vision Thread
# =========================================================

class ObjectDetectionYOLO(threading.Thread):
    """
    Real-time object detection and tracking thread:
      - reads camera frames
      - runs YOLO at a fixed rate
      - selects ONE target from detections
      - converts pixel coordinates to spatial angles
      - sends target updates to audio processor

    Produces single target with azimuth, elevation, distance, and confidence.
    """

    def __init__(self, processor):
        super().__init__(daemon=True)
        self.processor = processor
        self._stop_evt = threading.Event()
        self._frame_count = 0

    def stop(self):
        """Stop the vision thread."""
        self._stop_evt.set()

    def _compute_vfov_deg(self, hfov_deg: float, w: int, h: int) -> float:
        """Derive VFOV from HFOV + aspect ratio."""
        hf = math.radians(float(hfov_deg))
        vf = 2.0 * math.atan(math.tan(hf / 2.0) * (h / float(w)))
        return math.degrees(vf)

    def _estimate_distance_m(self, x1: float, y1: float, x2: float, y2: float, cls_name: str, frame_w: int, frame_h: int) -> float:
        """Estimate distance using bbox height + assumed real-world height (no depth AI)."""
        mode = VISION_CONFIG.get("distance_mode", "fixed")
        if mode == "fixed":
            return float(VISION_CONFIG.get("distance_fixed_m", 1.4))

        bbox_h = max(1.0, float(y2) - float(y1))

        sizes = VISION_CONFIG.get("class_real_heights_m", {}) or {}
        real_h = float(sizes.get(cls_name, sizes.get("person", 1.7)))

        hfov = float(VISION_CONFIG.get("hfov_deg", 70.0))
        vfov = self._compute_vfov_deg(hfov, frame_w, frame_h)

        # focal length in pixels (vertical)
        f = (frame_h / 2.0) / max(1e-6, math.tan(math.radians(vfov) / 2.0))

        dist = (real_h * f) / bbox_h

        dmin = float(VISION_CONFIG.get("distance_min_m", 0.3))
        dmax = float(VISION_CONFIG.get("distance_max_m", 6.0))
        dist = float(max(dmin, min(dmax, dist)))

        # smooth (EMA)
        if not hasattr(self, "_dist_ema"):
            self._dist_ema = dist
        alpha = float(VISION_CONFIG.get("distance_smoothing_alpha", 0.25))
        self._dist_ema = (1.0 - alpha) * self._dist_ema + alpha * dist
        return float(self._dist_ema)

    def run(self):
        """Main vision thread loop."""
        # Lazy imports so offline render can still run without these packages.
        import cv2
        from ultralytics import YOLO

        print("[VISION] Starting YOLO Phase-3 thread...")
        
        # Build full path to model file (relative to script location)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, VISION_CONFIG["model_path"])
        
        if not os.path.exists(model_path):
            print(f"[VISION][ERR] Model file not found: {model_path}")
            return
        
        model = YOLO(model_path)
        names = model.names

        cap = _open_camera_for_vision()
        if not cap.isOpened():
            print("[VISION][ERR] Could not open camera.")
            return

        _freeze_camera_settings(cap)

        infer_interval = 1.0 / float(VISION_CONFIG["infer_hz"])
        next_t = time.time()
        print_every = VISION_CONFIG.get("print_every_n_frames", 30)

        while not self._stop_evt.is_set():
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            now = time.time()
            if now < next_t:
                if VISION_CONFIG["show_window"]:
                    cv2.imshow(VISION_CONFIG["window_name"], frame)
                    if (cv2.waitKey(1) & 0xFF) == ord("q"):
                        self.stop()
                continue

            next_t = now + infer_interval
            self._frame_count += 1

            # YOLO inference with tracking
            results = model.track(frame, conf=VISION_CONFIG["conf_thres"], persist=True, verbose=False)
            
            if results is None or len(results) == 0:
                if VISION_CONFIG["show_window"]:
                    cv2.imshow(VISION_CONFIG["window_name"], frame)
                    if (cv2.waitKey(1) & 0xFF) == ord("q"):
                        self.stop()
                continue
            
            res = results[0]
            
            if res.boxes is None or len(res.boxes) == 0:
                if VISION_CONFIG["show_window"]:
                    cv2.imshow(VISION_CONFIG["window_name"], res.plot())
                    if (cv2.waitKey(1) & 0xFF) == ord("q"):
                        self.stop()
                continue

            # Process each tracked detection
            for detection in res.boxes:
                cls_id = int(detection.cls)
                cls_name = names.get(cls_id, str(cls_id))
                if cls_name != "person" or detection.id is None:
                    continue
                person_id = int(detection.id)
                source_id = person_id - 1  # ID 1 -> source 0, ID 2 -> source 1, etc.
                if source_id >= len(self.processor.sources):
                    continue

                x1, y1, x2, y2 = detection.xyxy[0].cpu().numpy()
                cx = 0.5 * (x1 + x2)
                cy = 0.5 * (y1 + y2)
                H, W = frame.shape[:2]

                az_deg = _pixels_to_azimuth_deg(cx, W, VISION_CONFIG["hfov_deg"])
                vfov_deg = self._compute_vfov_deg(VISION_CONFIG["hfov_deg"], W, H)
                ny = (cy - (H / 2.0)) / (H / 2.0)
                el_deg = -ny * (vfov_deg / 2.0)

                roll, pitch, yaw = self.processor.imu.get_euler()
                conf = float(detection.conf.cpu().numpy().item())
                dist_m = self._estimate_distance_m(x1, y1, x2, y2, cls_name, W, H)
                t_vision = time.time()

                self.processor.update_vision_target(
                    az_deg, el_deg, yaw_deg=yaw, pitch_deg=pitch,
                    distance_m=dist_m, conf=conf, cls_name=cls_name,
                    t_vision=t_vision, source_id=source_id
                )

                # Throttle prints
                if self._frame_count % print_every == 0:
                    print(f"[VISION] personID={person_id} source={source_id} conf={conf:.2f} az_deg={az_deg:.1f} el_deg={el_deg:.1f} dist_m={dist_m:.2f}")

            if VISION_CONFIG["show_window"]:
                annotated_frame = res.plot()
                cv2.imshow(VISION_CONFIG["window_name"], annotated_frame)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    self.stop()

        cap.release()
        if VISION_CONFIG["show_window"]:
            cv2.destroyWindow(VISION_CONFIG["window_name"])
        
        print("[VISION] Stopped.")

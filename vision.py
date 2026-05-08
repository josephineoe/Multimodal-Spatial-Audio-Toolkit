# ============================================================================
# VISION MODULE
# Real-time object detection and audio source control
# ============================================================================

import threading
import time
import os
import math

import numpy as np

# Import timing module for unified clock
from timing import system_clock


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

    # Detection mode (configurable via runtime key presses)
    # Mode 1: Both animate + inanimate objects
    # Mode 2: Inanimate objects only (bed, chair, couch, etc.)
    # Mode 3: Animate objects only (person) - RECOMMENDED FOR TESTING (IMU not ready)
    "detection_mode": 3,  # Start with person-only for testing

    # Class categories for multi-layer detection
    "animate_classes": {"person"},
    "inanimate_classes": {"cup", "chair", "couch", "bed", "dining table", "book", "microwave"},

    # Target selection mode for vision:
    #  - "allowed_objects": largest box among allowed_classes (includes person)
    #  - "person_only":     largest person box only
    "target_mode": "allowed_objects",

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
    try:
        import cv2
    except ImportError as e:
        print(f"[VISION][ERR] Failed to import cv2: {e}")
        return None
    
    src = VISION_CONFIG["camera_source"]
    try:
        if src == "default":
            # Try DirectShow backend first (more reliable on Windows)
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not cap.isOpened():
                print("[VISION] DirectShow failed, trying MSMF...")
                cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
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
    except Exception as e:
        print(f"[VISION][ERR] Failed to open camera: {e}")
        return None


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
        # ✓ FIXED: Use unified system clock for consistent timing
        self._last_detection_time = system_clock.now()  # Track last time a target was detected
        self._audio_playing = True  # Track if audio is currently active
        self._last_audio_stop_time = None  # Track when audio was last stopped to avoid spam
        # Object ID to source index mapping
        self._id_to_source = {}  # dict: {tracking_id -> source_idx}
        self._next_source_idx = 0  # Next available source index
        self._id_timeout_s = 1.0  # Remove mapping if object not seen for this long
        self._id_last_seen = {}  # dict: {tracking_id -> last_seen_time}

    def stop(self):
        """Stop the vision thread."""
        self._stop_evt.set()

    def _compute_vfov_deg(self, hfov_deg: float, w: int, h: int) -> float:
        """Derive VFOV from HFOV + aspect ratio."""
        hf = math.radians(float(hfov_deg))
        vf = 2.0 * math.atan(math.tan(hf / 2.0) * (h / float(w)))
        return math.degrees(vf)

    def _estimate_distance_m(self, x1: float, y1: float, x2: float, y2: float, cls_name: str, frame_w: int, frame_h: int) -> float:
        """
        Estimate distance using bbox height (not width!) and assumed real-world height.
        This function always uses the vertical size of the bounding box for distance estimation.
        """
        mode = VISION_CONFIG.get("distance_mode", "fixed")
        if mode == "fixed":
            return float(VISION_CONFIG.get("distance_fixed_m", 1.4))

        # Always use height (y2 - y1) for distance estimation
        bbox_h = max(1.0, float(y2) - float(y1))

        sizes = VISION_CONFIG.get("class_real_heights_m", {}) or {}
        real_h = float(sizes.get(cls_name, sizes.get("person", 1.7)))

        hfov = float(VISION_CONFIG.get("hfov_deg", 70.0))
        vfov = self._compute_vfov_deg(hfov, frame_w, frame_h)

        # Focal length in pixels (vertical)
        f = (frame_h / 2.0) / max(1e-6, math.tan(math.radians(vfov) / 2.0))

        dist = (real_h * f) / bbox_h

        dmin = float(VISION_CONFIG.get("distance_min_m", 0.3))
        dmax = float(VISION_CONFIG.get("distance_max_m", 6.0))
        dist = float(max(dmin, min(dmax, dist)))

        # Smooth (EMA)
        if not hasattr(self, "_dist_ema"):
            self._dist_ema = dist
        alpha = float(VISION_CONFIG.get("distance_smoothing_alpha", 0.25))
        self._dist_ema = (1.0 - alpha) * self._dist_ema + alpha * dist
        return float(self._dist_ema)

    def run(self):
        """Main vision thread loop."""
        # Lazy imports so offline render can still run without these packages.
        try:
            import cv2
        except ImportError as e:
            print(f"[VISION][ERR] Failed to import cv2: {e}")
            return
        
        # Check if cv2 has GUI support (imshow)
        if not hasattr(cv2, 'imshow'):
            print("[VISION][ERR] cv2 does not have GUI support (imshow).")
            print("           You may have cv2-headless installed instead of cv2.")
            print("           Install cv2 with: pip install opencv-python")
            return
        
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
        if cap is None or not cap.isOpened():
            print("[VISION][ERR] Could not open camera.")
            return

        _freeze_camera_settings(cap)

        # ✓ FIXED: Use unified system clock for consistent frame timing
        infer_interval = 1.0 / float(VISION_CONFIG["infer_hz"])
        next_t = system_clock.now()
        print_every = VISION_CONFIG.get("print_every_n_frames", 30)

        while not self._stop_evt.is_set():
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            # ✓ FIXED: Use unified system clock instead of wall-clock time
            now = system_clock.now()
            if now < next_t:
                # Skip to next inference time, don't display yet
                continue

            next_t = now + infer_interval
            self._frame_count += 1

            # YOLO inference with tracking
            results = model.track(frame, conf=VISION_CONFIG["conf_thres"], persist=True, verbose=False)
            
            # Track detection state for entire frame
            target_detected = False
            
            # Always process results if available
            if results is not None and len(results) > 0:
                res = results[0]
                
                if res.boxes is not None and len(res.boxes) > 0:
                    # ✓ FIXED: Process each tracked detection respecting detection_mode
                    # Mode 1: Both animate + inanimate
                    # Mode 2: Inanimate only
                    # Mode 3: Animate (person) only
                    
                    # Clean up stale object ID mappings (objects not seen recently)
                    current_time = system_clock.now()
                    stale_ids = [obj_id for obj_id, last_seen in self._id_last_seen.items()
                                 if (current_time - last_seen) > self._id_timeout_s]
                    for stale_id in stale_ids:
                        if stale_id in self._id_to_source:
                            source_idx = self._id_to_source[stale_id]
                            del self._id_to_source[stale_id]
                            del self._id_last_seen[stale_id]
                            print(f"[VISION] Deactivating source {source_idx} (object ID {stale_id} timeout)")
                    
                    for detection in res.boxes:
                        cls_id = int(detection.cls)
                        cls_name = names.get(cls_id, str(cls_id))
                        detection_mode = VISION_CONFIG.get("detection_mode", 3)
                        # Strict mode 3: only allow 'person' (case-insensitive)
                        if detection_mode == 3:
                            if cls_name.lower() != "person" or detection.id is None:
                                continue
                        elif detection_mode == 2:
                            inanimate_cls = {c.lower() for c in VISION_CONFIG.get("inanimate_classes", {})}
                            if cls_name.lower() not in inanimate_cls or detection.id is None:
                                continue
                        else:  # mode 1: both
                            animate_cls = {c.lower() for c in VISION_CONFIG.get("animate_classes", {"person"})}
                            inanimate_cls = {c.lower() for c in VISION_CONFIG.get("inanimate_classes", {})}
                            if cls_name.lower() not in (animate_cls | inanimate_cls) or detection.id is None:
                                continue
                        target_detected = True  # A target object was detected
                        
                        # Object ID to source mapping
                        obj_id = int(detection.id)
                        self._id_last_seen[obj_id] = current_time  # Mark as recently seen
                        
                        # Assign or reuse source index for this object ID
                        if obj_id not in self._id_to_source:
                            # New object: assign next available source
                            source_idx = self._next_source_idx % (len(self.processor.sources) or 1)
                            self._id_to_source[obj_id] = source_idx
                            self._next_source_idx = (self._next_source_idx + 1) % (len(self.processor.sources) or 1)
                            print(f"[VISION] Assigning object ID {obj_id} ({cls_name}) to source {source_idx}")
                        else:
                            source_idx = self._id_to_source[obj_id]
                        
                        person_id = obj_id  # Use object tracking ID
                        
                        # Get bounding box for this detection
                        x1, y1, x2, y2 = detection.xyxy[0].cpu().numpy()
                        cx = 0.5 * (x1 + x2)
                        H, W = frame.shape[:2]
                        
                        # Convert to azimuth and estimate distance
                        az_deg = _pixels_to_azimuth_deg(cx, W, VISION_CONFIG["hfov_deg"])
                        el_deg = 0.0  # azimuth-only for now
                        conf = float(detection.conf)
                        dist_m = self._estimate_distance_m(x1, y1, x2, y2, cls_name, W, H)
                        
                        # Get IMU angles for world-lock (skip if IMU disabled)
                        if self.processor is not None and getattr(self.processor, 'imu', None) is not None:
                            try:
                                roll, pitch, yaw = self.processor.imu.get_euler()
                            except Exception:
                                roll, pitch, yaw = 0.0, 0.0, 0.0
                        else:
                            roll, pitch, yaw = 0.0, 0.0, 0.0
                        
                        # Send update to audio processor
                        self.processor.update_vision_target(
                            az_deg, el_deg, yaw, pitch, distance_m=dist_m, conf=conf,
                            cls_name=cls_name, t_vision=system_clock.now(), source_id=source_idx
                        )
                        
                        print(f"[VISION] obj_id={person_id} source={source_idx} class={cls_name} conf={conf:.2f} az={az_deg:.1f}° dist={dist_m:.2f}m")
            
            # ✓ FIXED: Handle no-detection timeout (moved outside the branch)
            if target_detected:
                self._last_detection_time = system_clock.now()
                self._last_audio_stop_time = None  # Reset stop time tracking
                # Resume audio if it was stopped
                if not self._audio_playing and self.processor is not None:
                    try:
                        self.processor.start_playback()
                        self._audio_playing = True
                        print(f"[VISION] {len(self._id_to_source)} active tracked object(s) - resuming audio playback.")
                    except Exception as e:
                        print(f"[VISION] Could not resume playback: {e}")
            else:
                # Check if target detection timeout has expired
                no_detection_timeout = float(VISION_CONFIG.get("no_detection_fade_s", 0.75))
                time_since_last_detection = system_clock.elapsed_ms(self._last_detection_time) / 1000.0
                
                # Only stop audio once when timeout is exceeded
                if time_since_last_detection > no_detection_timeout and self._audio_playing and self.processor is not None:
                    # Avoid repeated stop calls
                    if self._last_audio_stop_time is None or (system_clock.now() - self._last_audio_stop_time) > 1.0:
                        try:
                            self.processor.stop_playback()
                            self._audio_playing = False
                            self._last_audio_stop_time = system_clock.now()
                            print(f"[VISION] No target detected for {time_since_last_detection:.2f}s (threshold: {no_detection_timeout}s) - stopping audio playback.")
                        except Exception as e:
                            print(f"[VISION] Could not stop playback: {e}")
                
                # Throttle debug prints for no detection
                if self._frame_count % (print_every * 2) == 0:
                    print(f"[VISION] No target detected - time since last: {time_since_last_detection:.2f}s (threshold: {no_detection_timeout}s)")

            # ✓ FIXED: Display window ONCE per frame (moved outside all branches)
            if VISION_CONFIG["show_window"]:
                if results is not None and len(results) > 0:
                    annotated_frame = results[0].plot()
                else:
                    annotated_frame = frame
                cv2.imshow(VISION_CONFIG["window_name"], annotated_frame)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    self.stop()

        cap.release()
        if VISION_CONFIG["show_window"]:
            cv2.destroyWindow(VISION_CONFIG["window_name"])
        
        print("[VISION] Stopped.")


def start_and_test_vision(processor):
    """
    Start vision thread and test object detection.
    Independently callable for testing purposes.
    """
    print("\n" + "=" * 70)
    print("VISION THREAD - START AND TEST")
    print("=" * 70)
    print()
    detection_mode = VISION_CONFIG.get("detection_mode", 3)
    mode_names = {1: "Both (animate+inanimate)", 2: "Inanimate only", 3: "Animate (person) only"}
    print("Vision Configuration:")
    print(f"  Detection Mode: {detection_mode} - {mode_names.get(detection_mode, 'Unknown')}")
    print(f"  Camera Source: {VISION_CONFIG['camera_source']}")
    print(f"  Resolution: {VISION_CONFIG['width']}x{VISION_CONFIG['height']}")
    print(f"  FPS: {VISION_CONFIG['fps']}")
    print(f"  Model: {VISION_CONFIG['model_path']}")
    print(f"  Confidence Threshold: {VISION_CONFIG['conf_thres']}")
    print(f"  Inference Rate: {VISION_CONFIG['infer_hz']} Hz")
    print("\n  Detection Classes:")
    print(f"    Animate: {VISION_CONFIG.get('animate_classes', set())}")
    print(f"    Inanimate: {VISION_CONFIG.get('inanimate_classes', set())}")
    print()
    
    try:
        vision_thread = ObjectDetectionYOLO(processor)
        vision_thread.start()
        print(f"[VISION] Thread started successfully: {vision_thread.is_alive()}")
        return vision_thread
    except Exception as e:
        print(f"[VISION] Error starting vision thread: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    """
    Test vision module independently.
    Checks camera access and YOLO model availability.
    """
    print("=" * 70)
    print("VISION MODULE TEST")
    print("=" * 70)
    print()
    print("Vision Configuration:")
    print(f"  Camera Source: {VISION_CONFIG['camera_source']}")
    print(f"  Resolution: {VISION_CONFIG['width']}x{VISION_CONFIG['height']}")
    print(f"  FPS: {VISION_CONFIG['fps']}")
    print(f"  Model: {VISION_CONFIG['model_path']}")
    print(f"  Confidence Threshold: {VISION_CONFIG['conf_thres']}")
    print(f"  Inference Rate: {VISION_CONFIG['infer_hz']} Hz")
    print()
    
    # Test camera access
    print("Testing camera access...")
    try:
        import cv2
        cap = _open_camera_for_vision()
        if cap.isOpened():
            print("✅ Camera opened successfully")
            _freeze_camera_settings(cap)
            # Try to read one frame
            ret, frame = cap.read()
            if ret:
                print(f"✅ Frame captured: {frame.shape}")
            else:
                print("⚠️  Could not read frame")
            cap.release()
        else:
            print("❌ Could not open camera")
    except Exception as e:
        print(f"❌ Camera test failed: {e}")
    
    # Test YOLO model
    print()
    print("Testing YOLO model...")
    try:
        from ultralytics import YOLO
        import os
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, VISION_CONFIG["model_path"])
        if os.path.exists(model_path):
            print(f"✅ Model file found: {model_path}")
            model = YOLO(model_path)
            print(f"✅ Model loaded successfully")
            print(f"   Classes: {len(model.names)}")
        else:
            print(f"❌ Model file not found: {model_path}")
    except Exception as e:
        print(f"❌ YOLO test failed: {e}")
    
    print()
    print("Vision module is ready for testing with main.py")


if __name__ == "__main__":
    """
    Test vision module independently.
    Checks camera access and YOLO model availability.
    """
    print("=" * 70)
    print("VISION MODULE TEST")
    print("=" * 70)
    print()
    print("Vision Configuration:")
    print(f"  Camera Source: {VISION_CONFIG['camera_source']}")
    print(f"  Resolution: {VISION_CONFIG['width']}x{VISION_CONFIG['height']}")
    print(f"  FPS: {VISION_CONFIG['fps']}")
    print(f"  Model: {VISION_CONFIG['model_path']}")
    print(f"  Confidence Threshold: {VISION_CONFIG['conf_thres']}")
    print(f"  Inference Rate: {VISION_CONFIG['infer_hz']} Hz")
    print()
    
    # Test camera access
    print("Testing camera access...")
    try:
        import cv2
        cap = _open_camera_for_vision()
        if cap.isOpened():
            print("✅ Camera opened successfully")
            _freeze_camera_settings(cap)
            # Try to read one frame
            ret, frame = cap.read()
            if ret:
                print(f"✅ Frame captured: {frame.shape}")
            else:
                print("⚠️  Could not read frame")
            cap.release()
        else:
            print("❌ Could not open camera")
    except Exception as e:
        print(f"❌ Camera test failed: {e}")
    
    # Test YOLO model
    print()
    print("Testing YOLO model...")
    try:
        from ultralytics import YOLO
        import os
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, VISION_CONFIG["model_path"])
        if os.path.exists(model_path):
            print(f"✅ Model file found: {model_path}")
            model = YOLO(model_path)
            print(f"✅ Model loaded successfully")
            print(f"   Classes: {len(model.names)}")
        else:
            print(f"❌ Model file not found: {model_path}")
    except Exception as e:
        print(f"❌ YOLO test failed: {e}")
    
    print()
    print("Vision module is ready for testing with main.py")

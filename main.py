# ============================================================================
# MULTIMODAL SPATIAL AUDIO TOOLKIT - PHASE 4
# Head-tracked HRTF spatial audio with real-time vision-driven source control
# ============================================================================

#notes: imu isnt logging
#the video did not record fully

import socket
import threading
import time
import csv
import os
import math

import numpy as np
import soundfile as sf
import sounddevice as sd
from scipy import signal

from dataclasses import dataclass


class DebugLogger:
    """Lightweight CSV logger (safe to call from background threads). Logging is disabled by default."""

    def __init__(self, debug_dir: str):
        os.makedirs(debug_dir, exist_ok=True)

        self._lock = threading.Lock()
        self._enabled = False
        self._debug_dir = debug_dir
        self._vision_fp = None
        self._vision_w = None
        self._imu_fp = None
        self._imu_w = None

    def enable(self):
        """Enable logging and open CSV files."""
        with self._lock:
            if self._enabled:
                return
            self._enabled = True
            try:
                self._vision_fp = open(os.path.join(self._debug_dir, "vision_log.csv"), "w", newline="")
                self._vision_w = csv.writer(self._vision_fp)
                self._vision_w.writerow(["timestamp", "az_cam_deg", "el_cam_deg", "dist_m", "conf", "cls_name", "latency_ms"])
                self._vision_fp.flush()

                self._imu_fp = open(os.path.join(self._debug_dir, "imu_log.csv"), "w", newline="")
                self._imu_w = csv.writer(self._imu_fp)
                self._imu_w.writerow(["timestamp", "qw", "qx", "qy", "qz", "roll_deg", "pitch_deg", "yaw_deg", "latency_ms"])
                self._imu_fp.flush()
            except Exception as e:
                print(f"[DEBUG] Error enabling logging: {e}")
                self._enabled = False

    def disable(self):
        """Disable logging and close CSV files."""
        with self._lock:
            if not self._enabled:
                return
            self._enabled = False
            try:
                if self._vision_fp:
                    self._vision_fp.close()
                if self._imu_fp:
                    self._imu_fp.close()
            except Exception:
                pass
            self._vision_fp = None
            self._vision_w = None
            self._imu_fp = None
            self._imu_w = None

    def log_vision(self, t, az_deg, el_deg, dist_m, conf, cls_name, t_vision):
        """Log vision data with latency calculation."""
        with self._lock:
            if not self._enabled or self._vision_w is None:
                return
            try:
                latency_ms = (time.time() - float(t_vision)) * 1000.0
                self._vision_w.writerow([float(t), float(az_deg), float(el_deg), float(dist_m),
                                         float(conf) if conf is not None else 0.0, str(cls_name) if cls_name else "", float(latency_ms)])
                self._vision_fp.flush()
            except Exception:
                pass

    def log_imu(self, t, qw, qx, qy, qz, roll, pitch, yaw, t_send):
        """Log IMU data with latency calculation."""
        with self._lock:
            if not self._enabled or self._imu_w is None:
                return
            try:
                latency_ms = (time.time() - float(t_send)) * 1000.0
                self._imu_w.writerow([float(t), float(qw), float(qx), float(qy), float(qz),
                                      float(roll), float(pitch), float(yaw), float(latency_ms)])
                self._imu_fp.flush()
            except Exception:
                pass

    def close(self):
        """Close all open files."""
        with self._lock:
            try:
                if self._vision_fp:
                    self._vision_fp.close()
            except Exception:
                pass
            try:
                if self._imu_fp:
                    self._imu_fp.close()
            except Exception:
                pass
            self._vision_fp = None
            self._vision_w = None
            self._imu_fp = None
            self._imu_w = None


@dataclass
class SourceState:
    """Single-source state (Phase 3): what vision wants the audio source to do."""
    azimuth_deg: float = 0.0
    elevation_deg: float = 0.0
    distance_est: float = 1.4
    gain: float = 1.0
    active: bool = False
    last_update_t: float = 0.0
    conf: float = 0.0
    cls_name: str = ""


# ----------------------------- VISION (Phase 3) -----------------------------
# Minimal vision->azimuth integration:
# - single target, single audio source
# - azimuth only (elevation forced to 0)
# - world-lock ON (handled inside update_vision_target)
# - no distance, no multi-source, no tracking IDs
#
# This is intentionally simple to avoid integration pain.

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
    "infer_hz": 8.0,            # FIXED: Reduced from 10 to 8 Hz for stability

    # Target selection mode for vision:
    #  - "allowed_objects": largest box among allowed_classes (includes person)
    #  - "person_only":     largest person box only
    "target_mode": "allowed_objects",

    # Allow-list (includes "person")
    "allowed_classes": {"cup", "chair", "couch", "bed", "dining table", "book", "microwave", "person"},

    # Phase 2 camera model (azimuth-only)
    "hfov_deg": 70.0,

    # ---------------- Phase 3 (audio-source mapping) ----------------
    # Gate / timeout
    "gate_conf_thres": 0.25,      # only update source if conf >= this
    "no_detection_fade_s": 0.75,  # if no valid target for this long -> fade out

    # Smoothing to reduce jitter (EMA): new = (1-b)*old + b*meas
    "smooth_beta_az": 0.20,
    "smooth_beta_el": 0.20,
    "smooth_beta_dist": 0.25,

    # Gain shaping from distance (simple): gain ~= ref/dist, clamped
    "distance_ref_m": 1.4,
    "gain_min": 0.3,  # Change from 0.0 to 0.3 (keeps audio at least 30% volume)
    "gain_max": 1.0,
    "gain_smooth_beta": 0.20,

    # Phase 2.5 distance (NO depth AI yet): estimate distance from bbox height using assumed real-world size.
    # Monocular approximation: distance ≈ (real_height_m * focal_length_px) / bbox_height_px
    # Works best for "person". For other classes it's a rough proxy.
    "distance_mode": "bbox_height",   # "bbox_height" | "fixed"
    "distance_fixed_m": 1.4,
    "distance_min_m": 0.3,
    "distance_max_m": 6.0,
    "distance_smoothing_alpha": 0.25,  # 0..1 (higher = faster response)
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

    # FIXED: Toggles for distance effects (debug-friendly)
    "enable_distance_attenuation": True,  # Controls distance-based loudness

    # Debug / UI
    "show_window": False,        # FIXED: Turn off during real runs
    "window_name": "Vision (YOLO Phase-2)",

    # FIXED: Print throttling
    "print_every_n_frames": 30,  # Only print vision updates every N frames
}


def _open_camera_for_vision():
    # Lazy import so offline render can still run even if OpenCV isn't installed.
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
    # nx in [-1, 1]
    nx = (cx - (W / 2.0)) / (W / 2.0)
    return nx * (hfov_deg / 2.0)


def _pick_target_index_xyxy(xyxy, cls_ids, names, mode, allowed_classes):
    """
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


class ObjectDetectionYOLO(threading.Thread):
    """
    Real-time object detection and tracking:
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
        self._frame_count = 0  # FIXED: For print throttling
        self.video_writer = None
        self.video_file = None

    def stop(self):
        self._stop_evt.set()
    
    def close_video_writer(self):
        """Close the video writer if it's open."""
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
            print(f"[VISION] FPV recording closed: {self.video_file}")

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
        # Lazy imports so offline render can still run without these packages.
        import cv2
        from ultralytics import YOLO

        print("[VISION] Starting YOLO Phase-2 thread...")
        
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
                    # Initialize video writer on first frame if debug window is enabled
                    if self.video_writer is None:
                        h, w = frame.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        self.video_file = os.path.join(os.path.dirname(__file__), "vision_fpv.mp4")
                        self.video_writer = cv2.VideoWriter(
                            self.video_file, fourcc, 30.0, (w, h)
                        )
                        print(f"[VISION] Recording FPV to: {self.video_file}")
                    
                    # Write frame to video
                    self.video_writer.write(frame)
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
                conf = float(detection.conf.cpu().numpy())
                dist_m = self._estimate_distance_m(x1, y1, x2, y2, cls_name, W, H)
                t_vision = time.time()

                self.processor.update_vision_target(
                    az_deg, el_deg, yaw_deg=yaw, pitch_deg=pitch,
                    distance_m=dist_m, conf=conf, cls_name=cls_name,
                    t_vision=t_vision, source_id=source_id
                )

                # Throttle prints - only print every N frames
                if self._frame_count % print_every == 0:
                    print(f"[VISION] personID={person_id} source={source_id} conf={conf:.2f} az_deg={az_deg:.1f} el_deg={el_deg:.1f} dist_m={dist_m:.2f}")

            if VISION_CONFIG["show_window"]:
                annotated_frame = res.plot()
                
                # Initialize video writer if not already done
                if self.video_writer is None:
                    h, w = annotated_frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    self.video_file = os.path.join(os.path.dirname(__file__), "vision_fpv.mp4")
                    self.video_writer = cv2.VideoWriter(
                        self.video_file, fourcc, 30.0, (w, h)
                    )
                    print(f"[VISION] Recording FPV to: {self.video_file}")
                
                # Write frame to video
                self.video_writer.write(annotated_frame)
                cv2.imshow(VISION_CONFIG["window_name"], annotated_frame)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    self.stop()

        cap.release()
        if VISION_CONFIG["show_window"]:
            cv2.destroyWindow(VISION_CONFIG["window_name"])
        
        # Close video writer if it was initialized
        if self.video_writer is not None:
            self.video_writer.release()
            print(f"[VISION] FPV recording saved: {self.video_file}")
        
        print("[VISION] Stopped.")


# =========================================================
# AudioSource: single mono audio stream + spatial params
# =========================================================

class SpatialAudioSource:
    """
    Represents a single spatial audio source with its own audio file and spatial position parameters.
    Handles audio playback, resampling, normalization, and position interpolation.
    """

    def __init__(self, audio_file, sample_rate, source_id,
                 azimuth=0.0, elevation=0.0, distance=1.4):
        self.source_id = source_id
        self.audio_file = audio_file

        # Load and process audio
        self.audio_data, file_sr = sf.read(audio_file)

        # Resample if needed
        if file_sr != sample_rate:
            self.audio_data = self._resample(self.audio_data, file_sr, sample_rate)

        # Convert to mono if stereo
        if self.audio_data.ndim > 1:
            self.audio_data = np.mean(self.audio_data, axis=1)

        # Normalize to -6dB to leave headroom (prevents clipping)
        max_amplitude = np.max(np.abs(self.audio_data))
        if max_amplitude > 0:
            self.audio_data = self.audio_data / max_amplitude * 0.5  # -6dB headroom

        # Spatial parameters (current and target for smooth interpolation)
        self.azimuth = float(azimuth)
        self.elevation = float(elevation)
        self.distance = float(distance)

        self.target_azimuth = float(azimuth)
        self.target_elevation = float(elevation)
        self.target_distance = float(distance)

        # Phase 3: per-source gain (lets vision adjust loudness without changing HRTF math)
        self.gain = 1.0
        self.target_gain = 1.0

        # Interpolation speed (degrees/samples for angles, m/samples for distance)
        transition_time_ms = 50.0
        samples_per_buffer = 4096
        updates_per_second = sample_rate / samples_per_buffer
        self.position_smoothing = np.exp(
            -1.0 / (transition_time_ms * 0.001 * updates_per_second)
        )

        # Playback state
        self.playback_position = 0
        self.is_active = True

        print(f"  Source {source_id}: Loaded '{audio_file}' "
              f"({len(self.audio_data) / sample_rate:.2f}s)")

    def _resample(self, data, orig_sr, target_sr):
        """Resample audio data to target sample rate."""
        num_samples = int(len(data) * target_sr / orig_sr)
        return signal.resample(data, num_samples)

    def smooth_position_update(self):
        """Smoothly interpolate current position toward target position."""
        s = self.position_smoothing
        self.azimuth = s * self.azimuth + (1.0 - s) * self.target_azimuth
        self.elevation = s * self.elevation + (1.0 - s) * self.target_elevation
        self.distance = s * self.distance + (1.0 - s) * self.target_distance
        self.gain = s * self.gain + (1.0 - s) * self.target_gain

    def set_target_position(self, azimuth=None, elevation=None, distance=None):
        """Set target position for smooth interpolation."""
        if azimuth is not None:
            # Wrap to [-180, 180]
            az = float(azimuth)
            self.target_azimuth = ((az + 180.0) % 360.0) - 180.0

        if elevation is not None:
            el = float(elevation)
            self.target_elevation = float(np.clip(el, -90.0, 90.0))

        if distance is not None:
            d = float(distance)
            self.target_distance = max(0.1, d)

    def set_target_gain(self, gain: float):
        """Set target gain for smooth interpolation (0..1 recommended)."""
        g = float(gain)
        self.target_gain = float(np.clip(g, 0.0, 1.0))

    def get_audio_chunk(self, frames):
        """Get next chunk of audio, looping if necessary."""
        start = self.playback_position
        end = start + frames

        # Loop audio
        if end > len(self.audio_data):
            chunk = np.concatenate([
                self.audio_data[start:],
                self.audio_data[:end - len(self.audio_data)]
            ])
            self.playback_position = end - len(self.audio_data)
        else:
            chunk = self.audio_data[start:end]
            self.playback_position = end

        # Pad if needed
        if len(chunk) < frames:
            chunk = np.pad(chunk, (0, frames - len(chunk)))

        if not self.is_active:
            return np.zeros(frames, dtype=np.float32)

        # Apply per-source gain before spatialization
        return (chunk.astype(np.float32) * float(self.gain))

    def reset_position(self):
        """Reset playback to beginning."""
        self.playback_position = 0


# =========================================================
# IMU / WORLD-LOCK SIGN CONVENTION
# =========================================================
# YAW_SIGN controls yaw direction convention throughout the system:
#   -1: rightward head turn maps to negative yaw (default)
#   +1: rightward head turn maps to positive yaw (flipped convention)
YAW_SIGN = -1


# =========================================================
# IMUReceiver: receives quaternions over UDP and exposes Euler
# =========================================================

class HeadTrackingReceiver:
    """
    Background UDP listener that receives head orientation quaternions and converts to Euler angles.
    Enables head-tracked audio by providing real-time roll, pitch, yaw orientation data.
    """

    def __init__(self, ip="0.0.0.0", port=5005, logger=None):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((ip, port))
        self.sock.setblocking(False)

        # Shared quaternion state
        self.qw = 1.0
        self.qx = 0.0
        self.qy = 0.0
        self.qz = 0.0

        self.t_send = 0.0
        self.logger = logger
        thread = threading.Thread(target=self._loop, daemon=True)
        thread.start()

        print(f"[IMU] Listening for quaternions on {ip}:{port}")

    def _loop(self):
        """Background UDP receive loop."""
        # Accept BOTH formats:
        #  1) CSV:          t_send,qw,qx,qy,qz
        #  2) Labeled text: qw: 0.919 qx: 0.057 qy: 0.275 qz: 0.275
        # The previous code only handled CSV and silently dropped everything else.
        while True:
            try:
                data, _ = self.sock.recvfrom(1024)
                s = data.decode(errors="ignore").strip()
                if not s:
                    continue

                # Format 1: CSV (preferred for latency logging)
                if "," in s:
                    parts = [p.strip() for p in s.split(",")]
                    if len(parts) >= 5:
                        t_send, qw, qx, qy, qz = map(float, parts[:5])
                    else:
                        # Unexpected CSV shape
                        continue
                else:
                    # Format 2: labeled text
                    # Example: "qw: 0.919 qx: 0.057 qy: 0.275 qz: 0.275"
                    tokens = s.replace(":", "").split()
                    if len(tokens) < 8:
                        continue
                    kv = dict(zip(tokens[0::2], tokens[1::2]))
                    qw = float(kv.get("qw"))
                    qx = float(kv.get("qx"))
                    qy = float(kv.get("qy"))
                    qz = float(kv.get("qz"))
                    # Sender did not provide a timestamp in this format
                    t_send = time.time()

                self.t_send = float(t_send)
                self.qw, self.qx, self.qy, self.qz = float(qw), float(qx), float(qy), float(qz)

                if self.logger:
                    roll, pitch, yaw = self.get_euler()
                    self.logger.log_imu(time.time(), qw, qx, qy, qz, roll, pitch, yaw, t_send)

            except Exception:
                # Ignore malformed/empty packets
                pass

    def get_euler(self):
        """
        Convert internal quaternion to roll, pitch, yaw (degrees).
        Uses standard aerospace convention (Z-Y-X).
        """
        w, x, y, z = self.qw, self.qx, self.qy, self.qz

        # Normalize quaternion
        n = np.sqrt(w * w + x * x + y * y + z * z)
        if n == 0.0:
            return 0.0, 0.0, 0.0
        w, x, y, z = w / n, x / n, y / n, z / n

        # roll (x-axis rotation)
        sinr = 2.0 * (w * x + y * z)
        cosr = 1.0 - 2.0 * (x * x + y * y)
        roll = np.degrees(np.arctan2(sinr, cosr))

        # pitch (y-axis rotation)
        sinp = 2.0 * (w * y - z * x)
        sinp_clamped = np.clip(sinp, -1.0, 1.0)
        pitch = np.degrees(np.arcsin(sinp_clamped))

        # yaw (z-axis rotation)
        siny = 2.0 * (w * z + x * y)
        cosy = 1.0 - 2.0 * (y * y + z * z)
        yaw = np.degrees(np.arctan2(siny, cosy))

        return float(roll), float(pitch), float(yaw)


# =========================================================
# MultiSourceHRTFAudio: HRTF engine + head-tracking + vision hook
# =========================================================

class SpatialAudioProcessor:
    def export_offline_render(self, duration_seconds=5, output_file=None):
        """
        Render a fixed-duration spatial audio output to a WAV file (offline, not real-time).
        Output file is simulated.wav, or simulated1.wav, simulated2.wav, etc. if file exists.
        """
        import os
        print(f"\n🎬 Offline rendering {duration_seconds}s of spatial audio...")

        # Reset sources + overlap
        for source in self.sources:
            source.reset_position()

        self.overlap_buffers = [
            np.zeros((self.hrir_length - 1, 2), dtype=np.float32)
            for _ in self.sources
        ]

        total_frames = int(duration_seconds * self.sample_rate)
        output_audio = []

        frames_per_block = self.buffer_size
        for block_start in range(0, total_frames, frames_per_block):
            block_len = min(frames_per_block, total_frames - block_start)
            mixed_output = np.zeros((block_len, 2), dtype=np.float32)
            for i, source in enumerate(self.sources):
                if source.is_active:
                    mono_block = source.get_audio_chunk(block_len)
                    spatial = self.spatialize_audio_block(mono_block, i)
                    mixed_output += spatial
            output_audio.append(mixed_output)

        full_audio = np.vstack(output_audio)

        # FIXED: Normalize ONCE at the end (not per source)
        max_val = float(np.max(np.abs(full_audio)))
        if max_val > 0.95:
            full_audio *= 0.95 / max_val

        if output_file is None:
            output_file = "simulated.wav"
            idx = 1
            while os.path.exists(output_file):
                output_file = f"simulated{idx}.wav"
                idx += 1
        sf.write(output_file, full_audio, self.sample_rate)
        print(f"✅ Offline render complete: {output_file}")
        print(f"   Duration: {duration_seconds:.2f}s")
        print(f"   Sample rate: {self.sample_rate}Hz")


    def __init__(self, audio_files, sofa_file, sample_rate=44100, imu_port=5005):
        self.sample_rate = sample_rate
        self.buffer_size = 4096

        # Throttled audio params print
        self._audio_print_counter = 0
        self._audio_print_interval = 50  # Print every N callbacks (~1 sec at 4096 buffer)

        # Debug logger (vision + IMU) - disabled by default, enabled when recording starts
        debug_dir = os.path.join(os.path.dirname(__file__), "debug_logs")
        os.makedirs(debug_dir, exist_ok=True)
        self.logger = DebugLogger(debug_dir)

        # Head tracking receiver
        self.imu = HeadTrackingReceiver(port=imu_port, logger=self.logger)

        # FIXED: Use ONE consistent yaw signal (raw, no filtering/gain here)
        # Filtering/gain applied only in audio_callback for head tracking
        self.filtered_yaw = 0.0
        self.filtered_pitch = 0.0

        # Sensitivity gains (applied in audio callback)
        self.yaw_gain = 2.0
        self.pitch_gain = 2.0

        # Phase 3: gating + fade out when detection disappears
        self.vision_timeout_s = float(VISION_CONFIG.get("no_detection_fade_s", 0.75))

        # Initialization gate: prevent audio output until IMU ready
        self._audio_ready = False

        # Load SOFA HRTF data
        print(f"Loading SOFA file: {sofa_file}")
        self.load_sofa_hrtf(sofa_file)

        self.hrir_length = self.hrir_left.shape[1]

        # Create audio sources
        print(f"\nInitializing {len(audio_files)} audio sources:")
        self.sources = []
        for i, audio_file in enumerate(audio_files):
            azimuth = (i - len(audio_files) // 2) * 40.0
            source = SpatialAudioSource(audio_file, sample_rate, i, azimuth=azimuth)
            self.sources.append(source)

        # Phase 3: multiple source states for multi-person tracking
        self.source_states = [SourceState(
            azimuth_deg=0.0,
            elevation_deg=0.0,
            distance_est=float(VISION_CONFIG.get("distance_fixed_m", 1.4)),
            gain=1.0,
            active=False,
        ) for _ in self.sources]

        self._source_states_lock = threading.Lock()

        # Phase 3: smoothed listener-relative controls for each vision source
        self._src_az_ema = [0.0] * len(self.sources)
        self._src_el_ema = [0.0] * len(self.sources)
        self._src_dist_ema = [float(VISION_CONFIG.get("distance_fixed_m", 1.4))] * len(self.sources)
        self._src_gain_ema = [1.0] * len(self.sources)
        self._vision_was_fresh = [False] * len(self.sources)

        self.prev_interp_hrir_left = [None] * len(self.sources)
        self.prev_interp_hrir_right = [None] * len(self.sources)
        self.hrir_smoothing = 0.92

        self.overlap_buffers = [
            np.zeros((self.hrir_length - 1, 2), dtype=np.float32)
            for _ in self.sources
        ]

        self.is_playing = False
        self.selected_source = 0

        self.is_recording = False
        self.recorded_frames = []
        self.recording_duration = 0.0

        print(f"\nInitialized {len(self.sources)} sources")
        for i, src in enumerate(self.sources):
            print(f"  Source {i}: Az={src.azimuth:.0f}°, El={src.elevation:.0f}°, Dist={src.distance:.2f}m")

    # ---------------- Vision API ----------------

    def update_vision_target(self, azimuth_deg, elevation_deg, yaw_deg, pitch_deg, distance_m=None, conf=None, cls_name=None, t_vision=None, source_id=0):
        """
        Provide a camera/head-relative vision target with timing alignment.
        yaw_deg/pitch_deg are RAW IMU angles (degrees) at the vision capture time.
        t_vision is the timestamp when vision captured this frame.

        If world-lock enabled:
          store target in world coords so if detection drops briefly, target stays stable.
        """
        t = t_vision if t_vision is not None else time.time()
        gate_th = float(VISION_CONFIG.get("gate_conf_thres", 0.25))
        c = float(conf) if conf is not None else 0.0
        cname = str(cls_name) if cls_name is not None else ""

        # Compute world coords
        az_w = float(azimuth_deg) + (YAW_SIGN * float(yaw_deg))
        el_w = float(elevation_deg) + float(pitch_deg)
        d = float(distance_m) if distance_m is not None else float(VISION_CONFIG.get("distance_fixed_m", 1.4))

        # Gain from confidence and (optionally) distance
        g_conf = float(np.clip(c / 0.8, 0.0, 1.0))
        if VISION_CONFIG.get("enable_distance_attenuation", True):
            ref_dist = float(VISION_CONFIG.get("distance_ref_m", 1.4))
            dist_factor = ref_dist / max(d, 0.3)
            dist_factor = float(np.clip(dist_factor, 0.5, 2.0))
        else:
            dist_factor = 1.0
        g_meas = g_conf * dist_factor
        g_meas = float(np.clip(g_meas, VISION_CONFIG.get("gain_min", 0.0), VISION_CONFIG.get("gain_max", 1.0)))

        with self._source_states_lock:
            ss = self.source_states[source_id]
            if c >= gate_th:
                ss.azimuth_deg = az_w
                ss.elevation_deg = el_w
                ss.distance_est = d
                ss.gain = g_meas
                ss.active = True
                ss.conf = c
                ss.cls_name = cname
                ss.last_update_t = t

        # Log vision data (off callback thread)
        try:
            self.logger.log_vision(t, azimuth_deg, elevation_deg, d, c, cname, t)
        except Exception:
            pass

    # ---------------- SOFA / HRTF support ----------------

    def load_sofa_hrtf(self, sofa_file):
        """Load HRTF data from SOFA file."""
        try:
            import sofar as sf_sofa
            sofa = sf_sofa.read_sofa(sofa_file)

            self.hrir_data = sofa.Data_IR
            self.source_positions = sofa.SourcePosition

            self.hrir_left = self.hrir_data[:, 0, :]
            self.hrir_right = self.hrir_data[:, 1, :]

            print(f"  Loaded {self.hrir_data.shape[0]} HRTF measurements")
            print(f"  HRIR length: {self.hrir_data.shape[2]} samples")

            self._parse_position_grid()
            self._compute_measurement_vectors()

        except ImportError:
            print("\n=== SOFA Library Not Found ===")
            print("Install: pip install sofar")
            raise
        except Exception as e:
            print(f"Error loading SOFA file: {e}")
            raise

    def _parse_position_grid(self):
        """Parse and organize the HRTF measurement positions."""
        self.azimuths = self.source_positions[:, 0]
        self.elevations = self.source_positions[:, 1]

        self.position_dict = {}
        for i, (az, el) in enumerate(zip(self.azimuths, self.elevations)):
            az_key = int(np.round(az))
            el_key = int(np.round(el))
            self.position_dict[(az_key, el_key)] = i

    def _compute_measurement_vectors(self):
        """Pre-compute direction unit vectors for interpolation."""
        az = np.radians(self.azimuths)
        el = np.radians(self.elevations)

        x = np.cos(el) * np.cos(az)
        y = np.cos(el) * np.sin(az)
        z = np.sin(el)

        self.measurement_vecs = np.vstack([x, y, z]).T

    def find_k_nearest_hrir_indices(self, azimuth, elevation, k=4):
        """Find K nearest measured HRTF directions on the sphere."""
        az_r = np.radians(azimuth)
        el_r = np.radians(elevation)

        q = np.array([
            np.cos(el_r) * np.cos(az_r),
            np.cos(el_r) * np.sin(az_r),
            np.sin(el_r)
        ])

        dots = self.measurement_vecs @ q
        k = min(k, len(dots))
        idx = np.argpartition(-dots, k - 1)[:k]
        idx = idx[np.argsort(-dots[idx])]
        return idx, dots[idx]

    def interpolate_hrir(self, azimuth, elevation, k=4, beta=12.0):
        """Interpolates HRIR using softmax weighting over nearest neighbors."""
        idx, cos_vals = self.find_k_nearest_hrir_indices(azimuth, elevation, k)

        exps = np.exp(beta * (cos_vals - np.max(cos_vals)))
        weights = exps / (np.sum(exps) + 1e-12)

        hrir_l = weights @ self.hrir_left[idx]
        hrir_r = weights @ self.hrir_right[idx]
        return hrir_l, hrir_r

    def apply_distance_attenuation(self, audio, distance):
        """
        Distance attenuation with toggle
        """
        if not VISION_CONFIG.get("enable_distance_attenuation", True):
            return audio

        ref_distance = 1.4
        attenuation = ref_distance / max(distance, 0.2)
        output = audio * min(attenuation, 3.0) * 0.6

        try:
            elevation = self.sources[self.selected_source].elevation
            elevation_factor = 1.0 + (elevation / 100.0)
            output *= float(np.clip(elevation_factor, 0.7, 1.3))
        except Exception:
            pass

        return output

    @staticmethod
    def _wrap_deg(x: float) -> float:
        return ((float(x) + 180.0) % 360.0) - 180.0

    def _ema_angle(self, old_deg: float, meas_deg: float, beta: float) -> float:
        """EMA for angles, handling wrap-around properly."""
        old = self._wrap_deg(old_deg)
        meas = self._wrap_deg(meas_deg)
        diff = self._wrap_deg(meas - old)
        return self._wrap_deg(old + float(beta) * diff)

    def spatialize_audio_block(self, mono_block, source_idx):
        """Apply HRTF convolution to audio block for a specific source."""
        source = self.sources[source_idx]
        source.smooth_position_update()

        hrir_l_new, hrir_r_new = self.interpolate_hrir(
            source.azimuth, source.elevation, k=4, beta=12.0
        )

        if self.prev_interp_hrir_left[source_idx] is None:
            self.prev_interp_hrir_left[source_idx] = hrir_l_new.copy()
            self.prev_interp_hrir_right[source_idx] = hrir_r_new.copy()

        s = self.hrir_smoothing
        hrir_l = s * self.prev_interp_hrir_left[source_idx] + (1.0 - s) * hrir_l_new
        hrir_r = s * self.prev_interp_hrir_right[source_idx] + (1.0 - s) * hrir_r_new

        self.prev_interp_hrir_left[source_idx] = hrir_l
        self.prev_interp_hrir_right[source_idx] = hrir_r

        conv_left = signal.fftconvolve(mono_block, hrir_l, mode='full')
        conv_right = signal.fftconvolve(mono_block, hrir_r, mode='full')

        output_length = len(mono_block)
        output = np.zeros((output_length, 2), dtype=np.float32)

        overlap_len = min(len(self.overlap_buffers[source_idx]), output_length)
        if overlap_len > 0:
            output[:overlap_len] = self.overlap_buffers[source_idx][:overlap_len]

        output[:, 0] += conv_left[:output_length]
        output[:, 1] += conv_right[:output_length]

        if len(conv_left) > output_length:
            self.overlap_buffers[source_idx] = np.column_stack([
                conv_left[output_length:],
                conv_right[output_length:]
            ])
        else:
            self.overlap_buffers[source_idx] = np.zeros((self.hrir_length - 1, 2), dtype=np.float32)

        # Distance encoding (choose ONE approach)
        output = self.apply_distance_attenuation(output, source.distance)
        # ...existing code...

        return output

    def audio_callback(self, outdata, frames, time_info, status):
        """
        Callback function with consistent yaw usage and final normalization only
        """
        # Gate: Don't output audio until IMU is initialized
        if not self._audio_ready:
            outdata[:] = np.zeros((frames, 2), dtype=np.float32)
            return

        if status:
            print(f"Audio status: {status}")

        # RAW IMU angles
        roll, pitch, yaw = self.imu.get_euler()
        
        # Apply YAW_SIGN for consistent convention throughout system
        yaw_signed = YAW_SIGN * yaw

        # Head-tracking mapping (single source default)
        target_yaw = self.yaw_gain * yaw_signed
        target_pitch = self.pitch_gain * pitch

        alpha = 0.3
        self.filtered_yaw = (1.0 - alpha) * self.filtered_yaw + alpha * target_yaw
        self.filtered_pitch = (1.0 - alpha) * self.filtered_pitch + alpha * target_pitch

        az_for_audio = self.filtered_yaw
        el_for_audio = self.filtered_pitch
        dist_for_audio = None
        gain_for_audio = 1.0 #already forced to be 1.0

        t_now = time.time()

        for i in range(len(self.sources)):
            with self._source_states_lock:
                ss = self.source_states[i]
                v_active = bool(ss.active)
                v_last_t = float(ss.last_update_t)
                v_az_w = float(ss.azimuth_deg)
                v_el_w = float(ss.elevation_deg)
                v_dist = float(ss.distance_est)
                v_gain = float(ss.gain)

            fresh = v_active and ((t_now - v_last_t) <= self.vision_timeout_s)

            if fresh:
                meas_az = self._wrap_deg(v_az_w - yaw_signed)
                meas_el = float(np.clip(v_el_w - float(pitch), -90.0, 90.0))
                meas_dist = float(v_dist)
                meas_gain = float(v_gain)

                if not self._vision_was_fresh[i]:
                    self._src_az_ema[i] = meas_az
                    self._src_el_ema[i] = meas_el
                    self._src_dist_ema[i] = meas_dist
                    self._src_gain_ema[i] = meas_gain
                    self._vision_was_fresh[i] = True

                b_az = float(VISION_CONFIG.get("smooth_beta_az", 0.20))
                b_el = float(VISION_CONFIG.get("smooth_beta_el", 0.20))
                b_d = float(VISION_CONFIG.get("smooth_beta_dist", 0.25))
                b_g = float(VISION_CONFIG.get("gain_smooth_beta", 0.20))

                self._src_az_ema[i] = self._ema_angle(self._src_az_ema[i], meas_az, b_az)
                self._src_el_ema[i] = (1.0 - b_el) * self._src_el_ema[i] + b_el * meas_el
                self._src_dist_ema[i] = (1.0 - b_d) * self._src_dist_ema[i] + b_d * meas_dist
                self._src_gain_ema[i] = (1.0 - b_g) * self._src_gain_ema[i] + b_g * meas_gain

            else:
                self._vision_was_fresh[i] = False
                b_g = float(VISION_CONFIG.get("gain_smooth_beta", 0.20))
                self._src_gain_ema[i] = (1.0 - b_g) * self._src_gain_ema[i]

            if self._src_gain_ema[i] > 0.02:
                az_for_audio = self._src_az_ema[i]
                el_for_audio = self._src_el_ema[i]
                dist_for_audio = self._src_dist_ema[i]
                gain_for_audio = float(np.clip(self._src_gain_ema[i], 0.0, 1.0))
            else:
                az_for_audio = self.filtered_yaw
                el_for_audio = self.filtered_pitch
                dist_for_audio = None
                gain_for_audio = 1.0

            self.sources[i].set_target_position(azimuth=az_for_audio, elevation=el_for_audio, distance=dist_for_audio)
            self.sources[i].set_target_gain(gain_for_audio)

        # Throttled debug print of audio parameters
        self._audio_print_counter += 1
        if self._audio_print_counter >= self._audio_print_interval:
            self._audio_print_counter = 0
            for i in range(len(self.sources)):
                ss = self.source_states[i]
                fresh = bool(ss.active) and ((t_now - float(ss.last_update_t)) <= self.vision_timeout_s)
                dist_for_audio = self._src_dist_ema[i] if self._src_gain_ema[i] > 0.02 else None
                gain_for_audio = float(np.clip(self._src_gain_ema[i], 0.0, 1.0)) if self._src_gain_ema[i] > 0.02 else 1.0
                az_for_audio = self._src_az_ema[i] if self._src_gain_ema[i] > 0.02 else self.filtered_yaw
                el_for_audio = self._src_el_ema[i] if self._src_gain_ema[i] > 0.02 else self.filtered_pitch
                dist_str = f"{dist_for_audio:.2f}m" if dist_for_audio is not None else "None"
                print(f"[AUDIO] source={i} az={az_for_audio:7.1f}° el={el_for_audio:6.1f}° dist={dist_str} gain={gain_for_audio:.2f} fresh={fresh}")

        mixed_output = np.zeros((frames, 2), dtype=np.float32)

        for i, source in enumerate(self.sources):
            if source.is_active:
                chunk = source.get_audio_chunk(frames)
                spatialized = self.spatialize_audio_block(chunk, i)
                mixed_output += spatialized

        max_val = float(np.max(np.abs(mixed_output)))
        if max_val > 0.95:
            mixed_output *= 0.95 / max_val

        if self.is_recording:
            self.recorded_frames.append(mixed_output.copy())
            self.recording_duration += frames / self.sample_rate

        outdata[:] = mixed_output

    def start_playback(self):
        """Start real-time audio playback with HRTF processing."""
        if self.is_playing:
            print("Already playing")
            return

        # Wait for IMU to send first packet before starting audio
        print("⏳ Waiting for IMU initialization...")
        timeout = time.time() + 3.0
        while self.imu.t_send == 0.0 and time.time() < timeout:
            time.sleep(0.05)
        
        if self.imu.t_send == 0.0:
            print("⚠️  WARNING: IMU not responding (starting anyway)")
        else:
            print("✅ IMU ready")
        
        # Mark audio as ready (enables audio_callback output)
        self._audio_ready = True
        self.is_playing = True

        for source in self.sources:
            source.reset_position()

        self.overlap_buffers = [
            np.zeros((self.hrir_length - 1, 2), dtype=np.float32)
            for _ in self.sources
        ]

        print("\n" + "=" * 60)
        print("HEAD-TRACKED HRTF SPATIAL AUDIO (QUATERNION-DRIVEN)")
        print("=" * 60)
        print("\nControls:")
        print("  • Move your head (IMU) → source 0 world-fixed by default.")
        print("  • If vision target is updated recently, source 0 follows vision.")
        print("  • Press Ctrl+C in the terminal to stop playback\n")
        print(f"\nDistance Attenuation: {'ON' if VISION_CONFIG.get('enable_distance_attenuation', False) else 'OFF'}")

        self.stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=2,
            callback=self.audio_callback,
            blocksize=self.buffer_size
        )
        self.stream.start()

    def stop_playback(self):
        """Stop audio playback."""
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()

        self.is_playing = False

        if self.is_recording:
            self.stop_recording()

        try:
            self.logger.close()
        except Exception:
            pass

        print("\nPlayback stopped")

    def start_recording(self):
        """Start recording the mixed spatial audio output and enable debug logging."""
        if not self.is_playing:
            print("Cannot record - playback not active")
            return

        self.is_recording = True
        self.recorded_frames = []
        self.recording_duration = 0.0
        
        # Enable debug logging when recording starts
        self.logger.enable()
        
        print("\n🔴 RECORDING STARTED")
        print("   Debug logging enabled (IMU and vision)")

    def stop_recording(self, filename="spatial_audio_output.wav"):
        """Stop recording and save to WAV file. Disable debug logging."""
        if not self.is_recording:
            return

        self.is_recording = False
        
        # Disable debug logging when recording stops
        self.logger.disable()

        if len(self.recorded_frames) == 0:
            print("No audio recorded")
            return

        full_audio = np.vstack(self.recorded_frames)
        sf.write(filename, full_audio, self.sample_rate)
        print(f"\n✅ Recording saved: {filename}")
        print(f"   Duration: {self.recording_duration:.2f} seconds")
        print(f"   Samples: {len(full_audio)}")
        print(f"   Channels: {full_audio.shape[1]}")

        self.recorded_frames = []
        self.recording_duration = 0.0

    def toggle_recording(self):
        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording()


if __name__ == "__main__":
    print("=" * 70)
    print("HRTF SPATIAL AUDIO PROCESSOR (HEAD-TRACKED, QUATERNIONS)")
    print("=" * 70)

    try:
        audio_files = ["rain.wav", "drums.wav"]

        processor = SpatialAudioProcessor(
            audio_files=audio_files,
            sofa_file="MIT_KEMAR_normal_pinna.sofa",
            sample_rate=44100,
            imu_port=5005
        )

        # Vision thread (YOLO) + simple CLI controls (recording toggle)
        vision_thread = None

        mode = input(
            "Choose mode:\n"
            "  1. Real-time playback (head-tracked)\n"
            "  2. Offline render to WAV (scripted animation)\n"
            "Choice (1/2): "
        ).strip()

        if mode == "2":
            duration = input("Offline render duration in seconds (default 5): ").strip()
            try:
                duration = float(duration)
            except Exception:
                duration = 5.0
            processor.export_offline_render(duration_seconds=duration)
        else:
            processor.start_playback()

            # Start vision thread (Phase 2.5). This was present in phase2.5 but missing here.
            processor.use_world_lock_for_vision = True
            vision_thread = ObjectDetectionYOLO(processor)
            vision_thread.start()
            print("[MAIN] Vision thread started:", vision_thread.is_alive())

            # Simple CLI controls in background:
            #  - 'r' + Enter: toggle recording
            #  - 'v' + Enter: toggle vision on/off
            #  - 'd' + Enter: toggle YOLO display (debug)
            #  - 'q' + Enter: quit
            ctl = {"vision": vision_thread}

            def _control_loop():
                while True:
                    try:
                        cmd = input("[CTRL] r=record, v=toggle vision, d=debug display, q=quit > ").strip().lower()
                    except Exception:
                        return
                    if cmd == "r":
                        processor.toggle_recording()
                    elif cmd == "d":
                        VISION_CONFIG["show_window"] = not VISION_CONFIG["show_window"]
                        state = "ON" if VISION_CONFIG["show_window"] else "OFF"
                        print(f"[CTRL] YOLO debug display: {state}")
                        if not VISION_CONFIG["show_window"]:
                            # Close video writer when debug window is turned off
                            try:
                                vt = ctl.get("vision")
                                if vt is not None and hasattr(vt, 'close_video_writer'):
                                    vt.close_video_writer()
                            except Exception:
                                pass
                    elif cmd == "v":
                        vt = ctl.get("vision")
                        if vt is None or not vt.is_alive():
                            try:
                                vt = ObjectDetectionYOLO(processor)
                                vt.start()
                                ctl["vision"] = vt
                                print("[CTRL] Vision started.")
                            except Exception as e:
                                print(f"[CTRL] Could not start vision: {e}")
                        else:
                            try:
                                vt.stop()
                                vt.close_video_writer()
                                vt.join(timeout=2.0)
                            except Exception:
                                pass
                            ctl["vision"] = None
                            print("[CTRL] Vision stopped.")
                    elif cmd == "q":
                        raise KeyboardInterrupt
                    else:
                        print("[CTRL] Unknown command.")

            threading.Thread(target=_control_loop, daemon=True).start()

            try:
                while True:
                    time.sleep(0.5)
            except KeyboardInterrupt:
                print("\n[MAIN] Stopping playback...")
            finally:
                # Stop vision thread if it was started
                try:
                    vt = None
                    try:
                        vt = locals().get('ctl', {}).get('vision', None)
                    except Exception:
                        vt = None
                    if vt is not None:
                        vt.close_video_writer()
                        vt.stop()
                        vt.join(timeout=2.0)
                except Exception:
                    pass
                # Ensure playback is stopped
                try:
                    processor.stop_playback()
                except Exception:
                    pass

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nMake sure you have the required audio files and SOFA file")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

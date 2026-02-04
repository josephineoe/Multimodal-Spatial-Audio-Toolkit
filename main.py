# ============================================================================
# MULTIMODAL SPATIAL AUDIO TOOLKIT - PHASE 3.5
# Head-tracked HRTF spatial audio with real-time vision-driven source control
# ============================================================================

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
    "model_path": "yolov8n.pt",
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
    "gain_min": 0.0,
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
    "enable_distance_modulation": False,   # Controls parking sensor effect (default off)

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

    def stop(self):
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
        # Lazy imports so offline render can still run without these packages.
        import cv2
        from ultralytics import YOLO

        print("[VISION] Starting YOLO Phase-2 thread...")
        model = YOLO(VISION_CONFIG["model_path"])
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

            # YOLO inference (no tracking IDs)
            res = model(frame, conf=VISION_CONFIG["conf_thres"], verbose=False)[0]
            boxes = res.boxes

            if boxes is None or len(boxes) == 0:
                if VISION_CONFIG["show_window"]:
                    cv2.imshow(VISION_CONFIG["window_name"], res.plot())
                    if (cv2.waitKey(1) & 0xFF) == ord("q"):
                        self.stop()
                continue

            xyxy = boxes.xyxy.cpu().numpy()
            cls_ids = boxes.cls.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()

            idx = _pick_target_index_xyxy(
                xyxy=xyxy,
                cls_ids=cls_ids,
                names=names,
                mode=VISION_CONFIG["target_mode"],
                allowed_classes=VISION_CONFIG["allowed_classes"],
            )

            if idx is None:
                if VISION_CONFIG["show_window"]:
                    cv2.imshow(VISION_CONFIG["window_name"], res.plot())
                    if (cv2.waitKey(1) & 0xFF) == ord("q"):
                        self.stop()
                continue

            x1, y1, x2, y2 = xyxy[idx]
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)
            H, W = frame.shape[:2]

            az_deg = _pixels_to_azimuth_deg(cx, W, VISION_CONFIG["hfov_deg"])
            # Elevation: map vertical position to degrees (center=0, up=+max, down=-max)
            vfov_deg = self._compute_vfov_deg(VISION_CONFIG["hfov_deg"], W, H)
            ny = (cy - (H / 2.0)) / (H / 2.0)  # -1 (top) to +1 (bottom)
            el_deg = -ny * (vfov_deg / 2.0)    # up is positive, down is negative

            # FIXED: Get RAW IMU angles for world-lock update (consistent yaw signal)
            roll, pitch, yaw = self.processor.imu.get_euler()

            cls_name = names.get(int(cls_ids[idx]), str(int(cls_ids[idx])))
            conf = float(confs[idx])

            # Estimate distance for the CURRENT single target (Phase 2.5)
            dist_m = self._estimate_distance_m(x1, y1, x2, y2, cls_name, W, H)

            # FIXED: Store timestamp for timing alignment
            t_vision = time.time()
            
            self.processor.update_vision_target(
                az_deg,
                el_deg,
                yaw_deg=yaw,
                pitch_deg=pitch,
                distance_m=dist_m,
                conf=conf,
                cls_name=cls_name,
                t_vision=t_vision,  # FIXED: Pass vision timestamp
            )

            # FIXED: Throttle prints - only print every N frames
            if self._frame_count % print_every == 0:
                print(f"[VISION] target={cls_name} conf={conf:.2f} az_deg={az_deg:.1f} el_deg={el_deg:.1f} dist_m={dist_m:.2f}")

            if VISION_CONFIG["show_window"]:
                cv2.imshow(VISION_CONFIG["window_name"], res.plot())
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    self.stop()

        cap.release()
        if VISION_CONFIG["show_window"]:
            cv2.destroyWindow(VISION_CONFIG["window_name"])
        print("[VISION] Stopped.")

# last edited 01/10/2026 working now trying to combine yolo
#
# NOTE ON TARGET RULES (keep this instruction)
# ---------------------------------------------------------
# When YOLO detects multiple things in a frame, you still need one thing to treat
# as the "target" (the thing you'll turn into a single audio source).
#
# "Largest box" is a simple tie-breaker / prioritization rule:
#   - Each detection has a bounding box area: (x2-x1) * (y2-y1)
#   - "Largest box" means: choose the detection with the biggest area.
# Why it's useful:
#   - Bigger box ≈ object is closer / more dominant in view
#   - Cheap heuristic early on
#   - Stable with tracking (BoT-SORT IDs) because the "main" object often stays "main"
#
# In the "allowed_objects" mode, you are NOT choosing "largest among everything".
# You are choosing "largest among your allow-list" (cup/chair/.../person).
# The other mode "person_only" forces the target to be a person.
# ---------------------------------------------------------


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
                    self.logger.log_imu(time.time(), qw, qx, qy, qz, roll, pitch, yaw)

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

    def apply_distance_modulation(self, audio, distance):
        """
        FIXED: Parking sensor modulation with toggle
        Faster pulses = closer object
        """
        if not VISION_CONFIG.get("enable_distance_modulation", False):
            return audio
            
        freq_close = 10.0  # Hz (close objects pulse fast)
        freq_far = 2.0     # Hz (far objects pulse slow)
        mod_depth = 0.4    # 40% modulation depth
        dist_clamped = float(np.clip(distance, 0.5, 6.0))
        t_norm = (6.0 - dist_clamped) / 5.5  # 0 = far, 1 = close
        mod_freq = freq_far + (freq_close - freq_far) * t_norm
        num_samples = len(audio)
        t = np.arange(num_samples) / self.sample_rate
        envelope = 1.0 + mod_depth * np.sin(2 * np.pi * mod_freq * t)
        if audio.ndim == 2:
            return audio * envelope[:, np.newaxis]
        else:
            return audio * envelope

    def __init__(self, audio_files, sofa_file, sample_rate=44100, imu_port=5005):
        self.sample_rate = sample_rate
        self.buffer_size = 4096

        # FIXED: Move latency logging off callback thread
        self.latency_log_queue = []  # Store latency data for batch writing
        debug_dir = os.path.join(os.path.dirname(__file__), "debug_logs")
        latency_csv_path = os.path.join(debug_dir, "latency_final_aar_audio.csv")
        self.latency_log = open(latency_csv_path, "w", newline="")
        self.latency_writer = csv.writer(self.latency_log)
        self.latency_writer.writerow(["timestamp", "latency_ms"])

        # Head tracking receiver
        self.imu = HeadTrackingReceiver(port=imu_port, logger=None)
        
        # FIXED: Use ONE consistent yaw signal (raw, no filtering/gain here)
        # Filtering/gain applied only in audio_callback for head tracking
        self.filtered_yaw = 0.0
        self.filtered_pitch = 0.0

        # Sensitivity gains (applied in audio callback)
        self.yaw_gain = 2.0
        self.pitch_gain = 2.0

        # Vision state (FIXED: belongs here, not AudioSource)
        self._vision_lock = threading.Lock()
        self._vision_az_cam = None
        self._vision_el_cam = None
        self._vision_t = 0.0
        self._vision_az_world = None
        self._vision_el_world = None
        self._vision_dist_m = None
        self._vision_conf = 0.0
        self._vision_cls_name = ""
        self.use_world_lock_for_vision = True
        # Phase 3: gating + fade out when detection disappears
        self.vision_timeout_s = float(VISION_CONFIG.get("no_detection_fade_s", 0.75))

        # Phase 3: single source state struct (room to expand to multiple later)
        self.source_state = SourceState(
            azimuth_deg=0.0,
            elevation_deg=0.0,
            distance_est=float(VISION_CONFIG.get("distance_fixed_m", 1.4)),
            gain=1.0,
            active=False,
        )

        # Phase 3: smoothed listener-relative controls for the single vision source
        self._src_az_ema = 0.0
        self._src_el_ema = 0.0
        self._src_dist_ema = float(VISION_CONFIG.get("distance_fixed_m", 1.4))
        self._src_gain_ema = 1.0
        self._vision_was_fresh = False

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

    def update_vision_target(self, azimuth_deg, elevation_deg, yaw_deg, pitch_deg, distance_m=None, conf=None, cls_name=None, t_vision=None):
        """
        FIXED: Provide a camera/head-relative vision target with timing alignment.
        yaw_deg/pitch_deg are RAW IMU angles (degrees) at the vision capture time.
        t_vision is the timestamp when vision captured this frame.

        If world-lock enabled:
          store target in world coords so if detection drops briefly, target stays stable.
        """
        t = t_vision if t_vision is not None else time.time()
        gate_th = float(VISION_CONFIG.get("gate_conf_thres", 0.25))
        c = float(conf) if conf is not None else 0.0
        cname = str(cls_name) if cls_name is not None else ""

        with self._vision_lock:
            self._vision_az_cam = float(azimuth_deg)
            self._vision_el_cam = float(elevation_deg)
            self._vision_t = t
            if distance_m is not None:
                self._vision_dist_m = float(distance_m)

            self._vision_conf = c
            self._vision_cls_name = cname

            # FIXED: Use RAW yaw (consistent signal everywhere)
            if self.use_world_lock_for_vision:
                # az_rel (head) = az_world - yaw  => az_world = az_rel + yaw
                self._vision_az_world = float(azimuth_deg) + float(yaw_deg)
                self._vision_el_world = float(elevation_deg) + float(pitch_deg)
            else:
                self._vision_az_world = None
                self._vision_el_world = None

            # Phase 3: update single-source state (MEASUREMENTS). Smoothing/fade happens in audio callback.
            # World coords are what we want to persist (so head motion reorients listener, not the world).
            if self.use_world_lock_for_vision and (self._vision_az_world is not None):
                az_w = float(self._vision_az_world)
                el_w = float(self._vision_el_world)
            else:
                az_w = float(self._vision_az_cam)
                el_w = float(self._vision_el_cam)

            d = float(self._vision_dist_m) if self._vision_dist_m is not None else float(VISION_CONFIG.get("distance_fixed_m", 1.4))

            # FIXED: g_meas derived from confidence (and optionally distance)
            # Base gain on confidence - higher confidence = higher gain
            g_conf = float(np.clip(c / 0.8, 0.0, 1.0))  # Normalize confidence to 0-1 range
            
            # Optional: Factor in distance (closer = louder, but capped)
            # This is SEPARATE from distance attenuation in the audio engine
            if VISION_CONFIG.get("enable_distance_attenuation", True):
                ref_dist = float(VISION_CONFIG.get("distance_ref_m", 1.4))
                dist_factor = ref_dist / max(d, 0.3)  # Closer = higher factor
                dist_factor = float(np.clip(dist_factor, 0.5, 2.0))  # Limit range
            else:
                dist_factor = 1.0
            
            # Combine confidence and distance factors
            g_meas = g_conf * dist_factor
            g_meas = float(np.clip(g_meas, 
                                  VISION_CONFIG.get("gain_min", 0.0), 
                                  VISION_CONFIG.get("gain_max", 1.0)))

            if c >= gate_th:
                self.source_state.azimuth_deg = az_w
                self.source_state.elevation_deg = el_w
                self.source_state.distance_est = d
                self.source_state.gain = g_meas
                self.source_state.active = True
                self.source_state.conf = c
                self.source_state.cls_name = cname
                self.source_state.last_update_t = t

        # FIXED: Log vision data (off callback thread)
        self.logger.log_vision(t, azimuth_deg, elevation_deg, d, c, cname)

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
        FIXED: Distance attenuation with toggle
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
        # shortest signed difference
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

        # FIXED: Robust block mixing from combined_aar.py (no per-source normalization)
        conv_left = signal.fftconvolve(mono_block, hrir_l, mode='full')
        conv_right = signal.fftconvolve(mono_block, hrir_r, mode='full')

        output_length = len(mono_block)
        output = np.zeros((output_length, 2), dtype=np.float32)

        overlap_len = min(len(self.overlap_buffers[source_idx]), output_length)
        if overlap_len > 0:
            output[:overlap_len] = self.overlap_buffers[source_idx][:overlap_len]

        output[:, 0] += conv_left[:output_length]
        output[:, 1] += conv_right[:output_length]

        # Save new overlap
        if len(conv_left) > output_length:
            self.overlap_buffers[source_idx] = np.column_stack([
                conv_left[output_length:],
                conv_right[output_length:]
            ])
        else:
            self.overlap_buffers[source_idx] = np.zeros((self.hrir_length - 1, 2), dtype=np.float32)

        # Apply distance effects (with toggles)
        output = self.apply_distance_attenuation(output, source.distance)
        output = self.apply_distance_modulation(output, source.distance)

        # FIXED: Removed per-source normalization - preserves distance-based loudness

        return output

    def audio_callback(self, outdata, frames, time_info, status):
        """
        FIXED: Callback function with consistent yaw usage and final normalization only
        """
        t_now = time.time()
        t_send = self.imu.t_send

        # FIXED: Queue latency data instead of writing immediately
        if t_send > 0:
            latency_ms = (t_now - t_send) * 1000.0
            self.latency_log_queue.append([t_now, latency_ms])
            
            # Batch write every 100 samples
            if len(self.latency_log_queue) >= 100:
                self.latency_writer.writerows(self.latency_log_queue)
                self.latency_log_queue.clear()

        if status:
            print(f"Audio status: {status}")

        # FIXED: Get RAW IMU angles (consistent yaw signal)
        roll, pitch, yaw = self.imu.get_euler()

        # FIXED: Apply gain and sign ONLY here (not mixed with raw elsewhere)
        target_yaw = -self.yaw_gain * yaw
        target_pitch = self.pitch_gain * pitch

        # Smooth the filtered values
        alpha = 0.3
        self.filtered_yaw = (1.0 - alpha) * self.filtered_yaw + alpha * target_yaw
        self.filtered_pitch = (1.0 - alpha) * self.filtered_pitch + alpha * target_pitch

        # Default (no vision): head-tracked, world-fixed sound (single source)
        az_for_audio = self.filtered_yaw
        el_for_audio = self.filtered_pitch
        dist_for_audio = None
        gain_for_audio = 1.0

        # Phase 3: vision drives a single source state (az/el/dist/gain), then IMU reorients it.
        with self._vision_lock:
            ss = self.source_state
            v_active = bool(ss.active)
            v_last_t = float(ss.last_update_t)
            v_az_w = float(ss.azimuth_deg)
            v_el_w = float(ss.elevation_deg)
            v_dist = float(ss.distance_est)
            v_gain = float(ss.gain)

        fresh = v_active and ((t_now - v_last_t) <= self.vision_timeout_s)

        if fresh:
            # FIXED: Convert world -> listener frame using RAW yaw (consistent)
            meas_az = self._wrap_deg(v_az_w - float(yaw))
            meas_el = float(np.clip(v_el_w - float(pitch), -90.0, 90.0))
            meas_dist = float(v_dist)
            meas_gain = float(v_gain)

            # If we just became "fresh" this frame, snap the EMAs to the measurement
            if not self._vision_was_fresh:
                self._src_az_ema = meas_az
                self._src_el_ema = meas_el
                self._src_dist_ema = meas_dist
                self._src_gain_ema = meas_gain
                self._vision_was_fresh = True

            b_az = float(VISION_CONFIG.get("smooth_beta_az", 0.20))
            b_el = float(VISION_CONFIG.get("smooth_beta_el", 0.20))
            b_d = float(VISION_CONFIG.get("smooth_beta_dist", 0.25))
            b_g = float(VISION_CONFIG.get("gain_smooth_beta", 0.20))

            self._src_az_ema = self._ema_angle(self._src_az_ema, meas_az, b_az)
            self._src_el_ema = (1.0 - b_el) * self._src_el_ema + b_el * meas_el
            self._src_dist_ema = (1.0 - b_d) * self._src_dist_ema + b_d * meas_dist
            self._src_gain_ema = (1.0 - b_g) * self._src_gain_ema + b_g * meas_gain

        else:
            # No valid detection recently -> fade out the vision-driven loudness
            self._vision_was_fresh = False
            b_g = float(VISION_CONFIG.get("gain_smooth_beta", 0.20))
            self._src_gain_ema = (1.0 - b_g) * self._src_gain_ema  # towards 0

        # If vision gain is meaningfully on, use vision-driven az/el/dist. Otherwise keep pure head tracking.
        if self._src_gain_ema > 0.02:
            az_for_audio = self._src_az_ema
            el_for_audio = self._src_el_ema
            dist_for_audio = self._src_dist_ema
            gain_for_audio = float(np.clip(self._src_gain_ema, 0.0, 1.0))

        if self.sources:
            self.sources[0].set_target_position(azimuth=az_for_audio, elevation=el_for_audio, distance=dist_for_audio)
            self.sources[0].set_target_gain(gain_for_audio)

        mixed_output = np.zeros((frames, 2), dtype=np.float32)

        for i, source in enumerate(self.sources):
            if source.is_active:
                chunk = source.get_audio_chunk(frames)
                spatialized = self.spatialize_audio_block(chunk, i)
                mixed_output += spatialized

        # FIXED: Normalize ONCE at the end (preserves distance-based loudness)
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
        print(f"\nDistance Attenuation: {'ON' if VISION_CONFIG.get('enable_distance_attenuation', True) else 'OFF'}")
        print(f"Distance Modulation: {'ON' if VISION_CONFIG.get('enable_distance_modulation', False) else 'OFF'}\n")

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
        
        # FIXED: Flush any remaining latency data
        if self.latency_log_queue:
            self.latency_writer.writerows(self.latency_log_queue)
            self.latency_log_queue.clear()
            
        try:
            self.latency_log.close()
        except Exception:
            pass

        self.is_playing = False

        if self.is_recording:
            self.stop_recording()

        print("\nPlayback stopped")

    def start_recording(self):
        """Start recording the mixed spatial audio output."""
        if not self.is_playing:
            print("Cannot record - playback not active")
            return

        self.is_recording = True
        self.recorded_frames = []
        self.recording_duration = 0.0
        print("\n🔴 RECORDING STARTED")

    def stop_recording(self, filename="spatial_audio_output.wav"):
        """Stop recording and save to WAV file."""
        if not self.is_recording:
            return

        self.is_recording = False

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


    # Offline rendering has been removed. Real-time recording is available via start_recording/stop_recording/toggle_recording.


if __name__ == "__main__":
    print("=" * 70)
    print("HRTF SPATIAL AUDIO PROCESSOR (HEAD-TRACKED, QUATERNIONS)")
    print("=" * 70)

    try:
        audio_files = ["rain.mp3"]

        processor = SpatialAudioProcessor(
            audio_files=audio_files,
            sofa_file="MIT_KEMAR_normal_pinna.sofa",
            sample_rate=44100,
            imu_port=5005
        )

        # Optional vision thread (may be started elsewhere); ensure the name exists for cleanup.
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
            try:
                while True:
                    time.sleep(0.5)
            except KeyboardInterrupt:
                print("\n[MAIN] Stopping playback...")
            finally:
                # Stop vision thread if it was started
                try:
                    if 'vision_thread' in globals() and vision_thread is not None:
                        vision_thread.stop()
                        vision_thread.join(timeout=2.0)
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
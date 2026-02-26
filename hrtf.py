# ============================================================================
# HRTF SPATIAL AUDIO MODULE
# Head-tracked HRTF spatial audio with IMU support
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


# =========================================================
# Debug Logger
# =========================================================

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


# =========================================================
# IMU Sign Convention
# =========================================================
YAW_SIGN = -1


# =========================================================
# HeadTrackingReceiver: IMU / Quaternion receiver
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
# SpatialAudioSource: audio source with spatial params
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

        # Phase 3: per-source gain
        self.gain = 1.0
        self.target_gain = 1.0

        # Interpolation speed
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
# SpatialAudioProcessor: HRTF engine with head tracking
# =========================================================

class SpatialAudioProcessor:
    """
    Real-time HRTF-based spatial audio processor with multiple simultaneous sources.
    Driven by IMU head-tracking (pitch/yaw) from quaternions.
    """

    def __init__(self, audio_files, sofa_file, sample_rate=44100, imu_port=5005, vision_config=None):
        self.sample_rate = sample_rate
        self.buffer_size = 4096
        self.vision_config = vision_config or {}

        # Throttled audio params print
        self._audio_print_counter = 0
        self._audio_print_interval = 50

        # Debug logger
        debug_dir = os.path.join(os.path.dirname(__file__), "debug_logs")
        os.makedirs(debug_dir, exist_ok=True)
        self.logger = DebugLogger(debug_dir)

        # Head tracking receiver
        self.imu = HeadTrackingReceiver(port=imu_port, logger=self.logger)

        # Yaw/pitch filtering
        self.filtered_yaw = 0.0
        self.filtered_pitch = 0.0

        # Sensitivity gains
        self.yaw_gain = 2.0
        self.pitch_gain = 2.0

        # Phase 3: Vision timeout
        self.vision_timeout_s = float(self.vision_config.get("no_detection_fade_s", 0.75))

        # Audio ready gate
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

        # Phase 3: source states
        self.source_states = [SourceState(
            azimuth_deg=0.0,
            elevation_deg=0.0,
            distance_est=float(self.vision_config.get("distance_fixed_m", 1.4)),
            gain=1.0,
            active=False,
        ) for _ in self.sources]

        self._source_states_lock = threading.Lock()

        # Phase 3: smoothed controls
        self._src_az_ema = [0.0] * len(self.sources)
        self._src_el_ema = [0.0] * len(self.sources)
        self._src_dist_ema = [float(self.vision_config.get("distance_fixed_m", 1.4))] * len(self.sources)
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

    # ---- Vision API ----

    def update_vision_target(self, azimuth_deg, elevation_deg, yaw_deg, pitch_deg, distance_m=None, conf=None, cls_name=None, t_vision=None, source_id=0):
        """
        Provide a camera/head-relative vision target with timing alignment.
        """
        t = t_vision if t_vision is not None else time.time()
        gate_th = float(self.vision_config.get("gate_conf_thres", 0.25))
        c = float(conf) if conf is not None else 0.0
        cname = str(cls_name) if cls_name is not None else ""

        # Compute world coords
        az_w = float(azimuth_deg) + (YAW_SIGN * float(yaw_deg))
        el_w = float(elevation_deg) + float(pitch_deg)
        d = float(distance_m) if distance_m is not None else float(self.vision_config.get("distance_fixed_m", 1.4))

        # Gain from confidence and distance
        g_conf = float(np.clip(c / 0.8, 0.0, 1.0))
        if self.vision_config.get("enable_distance_attenuation", True):
            ref_dist = float(self.vision_config.get("distance_ref_m", 1.4))
            dist_factor = ref_dist / max(d, 0.3)
            dist_factor = float(np.clip(dist_factor, 0.5, 2.0))
        else:
            dist_factor = 1.0
        g_meas = g_conf * dist_factor
        g_meas = float(np.clip(g_meas, self.vision_config.get("gain_min", 0.0), self.vision_config.get("gain_max", 1.0)))

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

        # Log vision data
        try:
            self.logger.log_vision(t, azimuth_deg, elevation_deg, d, c, cname, t)
        except Exception:
            pass

    # ---- SOFA / HRTF ----

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
        """Distance attenuation with toggle"""
        if not self.vision_config.get("enable_distance_attenuation", True):
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

        output = self.apply_distance_attenuation(output, source.distance)

        return output

    def audio_callback(self, outdata, frames, time_info, status):
        """Callback function with head tracking and vision control."""
        # Gate: Don't output audio until IMU is initialized
        if not self._audio_ready:
            outdata[:] = np.zeros((frames, 2), dtype=np.float32)
            return

        if status:
            print(f"Audio status: {status}")

        # RAW IMU angles
        roll, pitch, yaw = self.imu.get_euler()
        
        # Apply YAW_SIGN for consistent convention
        yaw_signed = YAW_SIGN * yaw

        # Head-tracking mapping
        target_yaw = self.yaw_gain * yaw_signed
        target_pitch = self.pitch_gain * pitch

        alpha = 0.3
        self.filtered_yaw = (1.0 - alpha) * self.filtered_yaw + alpha * target_yaw
        self.filtered_pitch = (1.0 - alpha) * self.filtered_pitch + alpha * target_pitch

        az_for_audio = self.filtered_yaw
        el_for_audio = self.filtered_pitch
        dist_for_audio = None
        gain_for_audio = 1.0

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

                b_az = float(self.vision_config.get("smooth_beta_az", 0.20))
                b_el = float(self.vision_config.get("smooth_beta_el", 0.20))
                b_d = float(self.vision_config.get("smooth_beta_dist", 0.25))
                b_g = float(self.vision_config.get("gain_smooth_beta", 0.20))

                self._src_az_ema[i] = self._ema_angle(self._src_az_ema[i], meas_az, b_az)
                self._src_el_ema[i] = (1.0 - b_el) * self._src_el_ema[i] + b_el * meas_el
                self._src_dist_ema[i] = (1.0 - b_d) * self._src_dist_ema[i] + b_d * meas_dist
                self._src_gain_ema[i] = (1.0 - b_g) * self._src_gain_ema[i] + b_g * meas_gain

            else:
                self._vision_was_fresh[i] = False
                b_g = float(self.vision_config.get("gain_smooth_beta", 0.20))
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

        # Throttled debug print
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
        
        # Mark audio as ready
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
        print(f"\nDistance Attenuation: {'ON' if self.vision_config.get('enable_distance_attenuation', False) else 'OFF'}")

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
        
        # Disable debug logging
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

    def export_offline_render(self, duration_seconds=5, output_file=None):
        """Render a fixed-duration spatial audio output to a WAV file."""
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

        # Normalize once at the end
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

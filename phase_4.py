import socket
import threading
import time
import csv
import math

import numpy as np
import soundfile as sf
import sounddevice as sd
from scipy import signal

from dataclasses import dataclass


@dataclass
class SourceState:
    """Per-object state: what vision wants each audio source to do."""
    azimuth_deg: float = 0.0
    elevation_deg: float = 0.0
    distance_est: float = 1.4
    gain: float = 1.0
    active: bool = False
    last_update_t: float = 0.0
    conf: float = 0.0
    cls_name: str = ""
    track_id: int = -1


# ----------------------------- VISION CONFIG -----------------------------

VISION_CONFIG = {
    "camera_source": "default",
    "gst_pipeline": "",
    "width": 1280,
    "height": 720,
    "fps": 30,
    "use_mjpeg": True,
    "model_path": "yolov8n.pt",
    "conf_thres": 0.25,
    "infer_hz": 10.0,
    "target_mode": "allowed_objects",
    "allowed_classes": {"cup", "chair", "couch", "bed", "dining table", "book", "microwave", "person"},
    "max_tracked_objects": 3,
    "hfov_deg": 70.0,
    "gate_conf_thres": 0.25,
    "no_detection_fade_s": 0.75,
    "smooth_beta_az": 0.20,
    "smooth_beta_el": 0.20,
    "smooth_beta_dist": 0.25,
    "gain_smooth_beta": 0.20,
    "distance_mode": "bbox_height",
    "distance_fixed_m": 1.4,
    "distance_min_m": 0.3,
    "distance_max_m": 6.0,
    "distance_smoothing_alpha": 0.25,
    "class_real_heights_m": {
        "person": 1.7, "chair": 1.0, "couch": 1.0, "bed": 0.6,
        "dining table": 0.75, "book": 0.25, "microwave": 0.35, "cup": 0.12,
    },
    "use_amplitude_modulation": True,
    "am_freq_close": 10.0,
    "am_freq_far": 2.0,
    "am_depth": 0.4,
    "show_window": True,
    "window_name": "Vision (YOLO Phase-4)",
}


def _open_camera_for_vision():
    import cv2
    src = VISION_CONFIG["camera_source"]
    if src == "default":
        return cv2.VideoCapture(0)
    elif src == "usb":
        return cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)
    elif src == "gstreamer":
        gst = VISION_CONFIG["gst_pipeline"]
        if not gst:
            raise ValueError("gst_pipeline is empty")
        return cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
    else:
        raise ValueError(f"Unknown camera_source: {src}")


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
    print(f"[VISION][CAM] {actual_w}x{actual_h}@{actual_fps:.2f}")


def _pixels_to_azimuth_deg(cx, W, hfov_deg):
    nx = (cx - (W / 2.0)) / (W / 2.0)
    return nx * (hfov_deg / 2.0)


class VisionYOLOMultiObject(threading.Thread):
    def __init__(self, processor):
        super().__init__(daemon=True)
        self.processor = processor
        self._stop_evt = threading.Event()
        self._dist_ema = {}

    def stop(self):
        self._stop_evt.set()

    def _compute_vfov_deg(self, hfov_deg, w, h):
        hf = math.radians(float(hfov_deg))
        vf = 2.0 * math.atan(math.tan(hf / 2.0) * (h / float(w)))
        return math.degrees(vf)

    def _estimate_distance_m(self, track_id, x1, y1, x2, y2, cls_name, frame_w, frame_h):
        mode = VISION_CONFIG.get("distance_mode", "fixed")
        if mode == "fixed":
            return float(VISION_CONFIG.get("distance_fixed_m", 1.4))
        bbox_h = max(1.0, float(y2) - float(y1))
        sizes = VISION_CONFIG.get("class_real_heights_m", {})
        real_h = float(sizes.get(cls_name, sizes.get("person", 1.7)))
        hfov = float(VISION_CONFIG.get("hfov_deg", 70.0))
        vfov = self._compute_vfov_deg(hfov, frame_w, frame_h)
        f = (frame_h / 2.0) / max(1e-6, math.tan(math.radians(vfov) / 2.0))
        dist = (real_h * f) / bbox_h
        dmin = float(VISION_CONFIG.get("distance_min_m", 0.3))
        dmax = float(VISION_CONFIG.get("distance_max_m", 6.0))
        dist = float(max(dmin, min(dmax, dist)))
        if track_id not in self._dist_ema:
            self._dist_ema[track_id] = dist
        alpha = float(VISION_CONFIG.get("distance_smoothing_alpha", 0.25))
        self._dist_ema[track_id] = (1.0 - alpha) * self._dist_ema[track_id] + alpha * dist
        return float(self._dist_ema[track_id])

    def run(self):
        import cv2
        from ultralytics import YOLO
        print("[VISION] Starting Phase-4 (multi-object tracking)...")
        model = YOLO(VISION_CONFIG["model_path"])
        names = model.names
        cap = _open_camera_for_vision()
        if not cap.isOpened():
            print("[VISION][ERR] Could not open camera.")
            return
        _freeze_camera_settings(cap)
        infer_interval = 1.0 / float(VISION_CONFIG["infer_hz"])
        next_t = time.time()

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
            res = model.track(frame, conf=VISION_CONFIG["conf_thres"], 
                             persist=True, tracker="botsort.yaml", verbose=False)[0]
            boxes = res.boxes
            if boxes is None or len(boxes) == 0:
                if VISION_CONFIG["show_window"]:
                    cv2.imshow(VISION_CONFIG["window_name"], res.plot())
                    if (cv2.waitKey(1) & 0xFF) == ord("q"):
                        self.stop()
                continue
            roll, pitch, yaw = self.processor.imu.get_euler()
            H, W = frame.shape[:2]
            if boxes.id is not None:
                track_ids = boxes.id.int().cpu().tolist()
                xyxy = boxes.xyxy.cpu().numpy()
                cls_ids = boxes.cls.cpu().numpy().astype(int)
                confs = boxes.conf.cpu().numpy()
                allowed = VISION_CONFIG["allowed_classes"]
                max_objects = VISION_CONFIG.get("max_tracked_objects", 3)
                valid_tracks = []
                for i, track_id in enumerate(track_ids):
                    cls_name = names.get(int(cls_ids[i]), str(int(cls_ids[i])))
                    if cls_name not in allowed:
                        continue
                    x1, y1, x2, y2 = xyxy[i]
                    area = (x2 - x1) * (y2 - y1)
                    valid_tracks.append((track_id, i, area, cls_name))
                valid_tracks.sort(key=lambda x: x[2], reverse=True)
                valid_tracks = valid_tracks[:max_objects]
                for track_id, idx, area, cls_name in valid_tracks:
                    x1, y1, x2, y2 = xyxy[idx]
                    cx = 0.5 * (x1 + x2)
                    conf = float(confs[idx])
                    az_deg = _pixels_to_azimuth_deg(cx, W, VISION_CONFIG["hfov_deg"])
                    el_deg = 0.0
                    dist_m = self._estimate_distance_m(track_id, x1, y1, x2, y2, cls_name, W, H)
                    self.processor.update_tracked_object(
                        track_id=track_id, azimuth_deg=az_deg, elevation_deg=el_deg,
                        distance_m=dist_m, conf=conf, cls_name=cls_name,
                        yaw_deg=yaw, pitch_deg=pitch
                    )
                    print(f"[VISION] ID={track_id} {cls_name} conf={conf:.2f} az={az_deg:.1f}° dist={dist_m:.2f}m")
            if VISION_CONFIG["show_window"]:
                cv2.imshow(VISION_CONFIG["window_name"], res.plot())
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    self.stop()
        cap.release()
        if VISION_CONFIG["show_window"]:
            cv2.destroyWindow(VISION_CONFIG["window_name"])
        print("[VISION] Stopped.")


class AudioSource:
    def __init__(self, audio_file, sample_rate, source_id, azimuth=0.0, elevation=0.0, distance=1.4):
        self.source_id = source_id
        self.audio_file = audio_file
        self.audio_data, file_sr = sf.read(audio_file)
        if file_sr != sample_rate:
            num_samples = int(len(self.audio_data) * sample_rate / file_sr)
            self.audio_data = signal.resample(self.audio_data, num_samples)
        if self.audio_data.ndim > 1:
            self.audio_data = np.mean(self.audio_data, axis=1)
        max_amplitude = np.max(np.abs(self.audio_data))
        if max_amplitude > 0:
            self.audio_data = self.audio_data / max_amplitude * 0.5
        self.azimuth = float(azimuth)
        self.elevation = float(elevation)
        self.distance = float(distance)
        self.target_azimuth = float(azimuth)
        self.target_elevation = float(elevation)
        self.target_distance = float(distance)
        self.gain = 1.0
        self.target_gain = 1.0
        transition_time_ms = 50.0
        samples_per_buffer = 4096
        updates_per_second = sample_rate / samples_per_buffer
        self.position_smoothing = np.exp(-1.0 / (transition_time_ms * 0.001 * updates_per_second))
        self.playback_position = 0
        self.is_active = True
        print(f"  Source {source_id}: Loaded '{audio_file}' ({len(self.audio_data) / sample_rate:.2f}s)")

    def smooth_position_update(self):
        s = self.position_smoothing
        self.azimuth = s * self.azimuth + (1.0 - s) * self.target_azimuth
        self.elevation = s * self.elevation + (1.0 - s) * self.target_elevation
        self.distance = s * self.distance + (1.0 - s) * self.target_distance
        self.gain = s * self.gain + (1.0 - s) * self.target_gain

    def set_target_position(self, azimuth=None, elevation=None, distance=None):
        if azimuth is not None:
            az = float(azimuth)
            self.target_azimuth = ((az + 180.0) % 360.0) - 180.0
        if elevation is not None:
            self.target_elevation = float(np.clip(elevation, -90.0, 90.0))
        if distance is not None:
            self.target_distance = max(0.1, float(distance))

    def set_target_gain(self, gain):
        self.target_gain = float(np.clip(gain, 0.0, 1.0))

    def get_audio_chunk(self, frames):
        start = self.playback_position
        end = start + frames
        if end > len(self.audio_data):
            chunk = np.concatenate([self.audio_data[start:], self.audio_data[:end - len(self.audio_data)]])
            self.playback_position = end - len(self.audio_data)
        else:
            chunk = self.audio_data[start:end]
            self.playback_position = end
        if len(chunk) < frames:
            chunk = np.pad(chunk, (0, frames - len(chunk)))
        if not self.is_active:
            return np.zeros(frames, dtype=np.float32)
        return chunk.astype(np.float32) * float(self.gain)

    def reset_position(self):
        self.playback_position = 0


class IMUReceiver:
    def __init__(self, ip="0.0.0.0", port=5005):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((ip, port))
        self.sock.setblocking(False)
        self.qw = 1.0
        self.qx = 0.0
        self.qy = 0.0
        self.qz = 0.0
        self.t_send = 0.0
        thread = threading.Thread(target=self._loop, daemon=True)
        thread.start()
        print(f"[IMU] Listening on {ip}:{port}")

    def _loop(self):
        while True:
            try:
                data, _ = self.sock.recvfrom(1024)
                s = data.decode(errors="ignore").strip()
                if not s:
                    continue
                if "," in s:
                    parts = [p.strip() for p in s.split(",")]
                    if len(parts) >= 5:
                        t_send, qw, qx, qy, qz = map(float, parts[:5])
                    else:
                        continue
                else:
                    tokens = s.replace(":", "").split()
                    if len(tokens) < 8:
                        continue
                    kv = dict(zip(tokens[0::2], tokens[1::2]))
                    qw = float(kv.get("qw"))
                    qx = float(kv.get("qx"))
                    qy = float(kv.get("qy"))
                    qz = float(kv.get("qz"))
                    t_send = time.time()
                self.t_send = float(t_send)
                self.qw, self.qx, self.qy, self.qz = float(qw), float(qx), float(qy), float(qz)
            except Exception:
                pass

    def get_euler(self):
        w, x, y, z = self.qw, self.qx, self.qy, self.qz
        n = np.sqrt(w*w + x*x + y*y + z*z)
        if n == 0.0:
            return 0.0, 0.0, 0.0
        w, x, y, z = w/n, x/n, y/n, z/n
        sinr = 2.0 * (w*x + y*z)
        cosr = 1.0 - 2.0 * (x*x + y*y)
        roll = np.degrees(np.arctan2(sinr, cosr))
        sinp = 2.0 * (w*y - z*x)
        pitch = np.degrees(np.arcsin(np.clip(sinp, -1.0, 1.0)))
        siny = 2.0 * (w*z + x*y)
        cosy = 1.0 - 2.0 * (y*y + z*z)
        yaw = np.degrees(np.arctan2(siny, cosy))
        return float(roll), float(pitch), float(yaw)


class MultiSourceHRTFAudio:
    def __init__(self, audio_files, sofa_file, sample_rate=44100, imu_port=5005):
        self.sample_rate = sample_rate
        self.buffer_size = 4096
        self.latency_log = open("latency_phase4.csv", "w", newline="")
        self.latency_writer = csv.writer(self.latency_log)
        self.latency_writer.writerow(["timestamp", "latency_ms"])
        self.imu = IMUReceiver(port=imu_port)
        self.filtered_yaw = 0.0
        self.filtered_pitch = 0.0
        self.yaw_gain = 2.0
        self.pitch_gain = 2.0
        self._vision_lock = threading.Lock()
        self.source_states = {}
        self.track_to_source_idx = {}
        self.max_active_sources = VISION_CONFIG.get("max_tracked_objects", 3)
        self._src_az_ema = {}
        self._src_el_ema = {}
        self._src_dist_ema = {}
        self._src_gain_ema = {}
        self._vision_was_fresh = {}
        self.use_world_lock_for_vision = True
        self.vision_timeout_s = float(VISION_CONFIG.get("no_detection_fade_s", 0.75))
        print(f"Loading SOFA file: {sofa_file}")
        self.load_sofa_hrtf(sofa_file)
        self.hrir_length = self.hrir_left.shape[1]
        print(f"\nInitializing {len(audio_files)} audio sources:")
        self.sources = []
        for i, audio_file in enumerate(audio_files):
            source = AudioSource(audio_file, sample_rate, i, azimuth=0.0)
            self.sources.append(source)
        self.prev_interp_hrir_left = [None] * len(self.sources)
        self.prev_interp_hrir_right = [None] * len(self.sources)
        self.hrir_smoothing = 0.92
        self.overlap_buffers = [np.zeros((self.hrir_length - 1, 2), dtype=np.float32) for _ in self.sources]
        self.is_playing = False
        self.is_recording = False
        self.recorded_frames = []
        self.recording_duration = 0.0

    def update_tracked_object(self, track_id, azimuth_deg, elevation_deg, distance_m, conf, cls_name, yaw_deg, pitch_deg):
        t = time.time()
        gate_th = float(VISION_CONFIG.get("gate_conf_thres", 0.25))
        with self._vision_lock:
            if track_id not in self.source_states:
                self.source_states[track_id] = SourceState(track_id=track_id)
            ss = self.source_states[track_id]
            if self.use_world_lock_for_vision:
                az_w = float(azimuth_deg) + float(yaw_deg)
                el_w = float(elevation_deg) + float(pitch_deg)
            else:
                az_w = float(azimuth_deg)
                el_w = float(elevation_deg)
            if conf >= gate_th:
                ss.azimuth_deg = az_w
                ss.elevation_deg = el_w
                ss.distance_est = float(distance_m)
                ss.gain = 1.0
                ss.active = True
                ss.conf = float(conf)
                ss.cls_name = str(cls_name)
                ss.last_update_t = t

    def _allocate_audio_source(self, track_id):
        if track_id in self.track_to_source_idx:
            return self.track_to_source_idx[track_id]
        for i in range(min(len(self.sources), self.max_active_sources)):
            if i not in self.track_to_source_idx.values():
                self.track_to_source_idx[track_id] = i
                return i
        oldest_track = min(self.source_states.items(), key=lambda x: x[1].last_update_t if x[1].active else 0)[0]
        idx = self.track_to_source_idx.pop(oldest_track)
        self.track_to_source_idx[track_id] = idx
        return idx

    def load_sofa_hrtf(self, sofa_file):
        try:
            import sofar as sf_sofa
            sofa = sf_sofa.read_sofa(sofa_file)
            self.hrir_data = sofa.Data_IR
            self.source_positions = sofa.SourcePosition
            self.hrir_left = self.hrir_data[:, 0, :]
            self.hrir_right = self.hrir_data[:, 1, :]
            print(f"  Loaded {self.hrir_data.shape[0]} HRTF measurements")
            self._parse_position_grid()
            self._compute_measurement_vectors()
        except ImportError:
            print("\n=== SOFA Library Not Found ===\nInstall: pip install sofar")
            raise

    def _parse_position_grid(self):
        self.azimuths = self.source_positions[:, 0]
        self.elevations = self.source_positions[:, 1]
        self.position_dict = {}
        for i, (az, el) in enumerate(zip(self.azimuths, self.elevations)):
            self.position_dict[(int(np.round(az)), int(np.round(el)))] = i

    def _compute_measurement_vectors(self):
        az = np.radians(self.azimuths)
        el = np.radians(self.elevations)
        x = np.cos(el) * np.cos(az)
        y = np.cos(el) * np.sin(az)
        z = np.sin(el)
        self.measurement_vecs = np.vstack([x, y, z]).T

    def find_k_nearest_hrir_indices(self, azimuth, elevation, k=4):
        az_r = np.radians(azimuth)
        el_r = np.radians(elevation)
        q = np.array([np.cos(el_r)*np.cos(az_r), np.cos(el_r)*np.sin(az_r), np.sin(el_r)])
        dots = self.measurement_vecs @ q
        k = min(k, len(dots))
        idx = np.argpartition(-dots, k-1)[:k]
        idx = idx[np.argsort(-dots[idx])]
        return idx, dots[idx]

    def interpolate_hrir(self, azimuth, elevation, k=4, beta=12.0):
        idx, cos_vals = self.find_k_nearest_hrir_indices(azimuth, elevation, k)
        exps = np.exp(beta * (cos_vals - np.max(cos_vals)))
        weights = exps / (np.sum(exps) + 1e-12)
        return weights @ self.hrir_left[idx], weights @ self.hrir_right[idx]

    def apply_distance_modulation(self, audio, distance):
        if not VISION_CONFIG.get("use_amplitude_modulation", True):
            return audio
        freq_close = float(VISION_CONFIG.get("am_freq_close", 10.0))
        freq_far = float(VISION_CONFIG.get("am_freq_far", 2.0))
        mod_depth = float(VISION_CONFIG.get("am_depth", 0.4))
        dist_clamped = float(np.clip(distance, 0.5, 6.0))
        t_norm = (6.0 - dist_clamped) / 5.5
        mod_freq = freq_far + (freq_close - freq_far) * t_norm
        t = np.arange(len(audio)) / self.sample_rate
        envelope = 1.0 + mod_depth * np.sin(2 * np.pi * mod_freq * t)
        return audio * envelope[:, np.newaxis] if audio.ndim == 2 else audio * envelope

    def apply_distance_attenuation(self, audio, distance):
        ref_distance = 1.4
        attenuation = ref_distance / max(distance, 0.2)
        return audio * min(attenuation, 3.0) * 0.6

    @staticmethod
    def _wrap_deg(x):
        return ((float(x) + 180.0) % 360.0) - 180.0

    def _ema_angle(self, old_deg, meas_deg, beta):
        old = self._wrap_deg(old_deg)
        meas = self._wrap_deg(meas_deg)
        diff = self._wrap_deg(meas - old)
        return self._wrap_deg(old + float(beta) * diff)

    def spatialize_audio_block(self, mono_block, source_idx):
        source = self.sources[source_idx]
        source.smooth_position_update()
        hrir_l_new, hrir_r_new = self.interpolate_hrir(source.azimuth, source.elevation, k=4, beta=12.0)
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
            self.overlap_buffers[source_idx] = np.column_stack([conv_left[output_length:], conv_right[output_length:]])
        else:
            self.overlap_buffers[source_idx] = np.zeros((self.hrir_length - 1, 2), dtype=np.float32)
        output = self.apply_distance_attenuation(output, source.distance)
        output = self.apply_distance_modulation(output, source.distance)
        return output

    def audio_callback(self, outdata, frames, time_info, status):
        t_now = time.time()
        if self.imu.t_send > 0:
            self.latency_writer.writerow([t_now, (t_now - self.imu.t_send) * 1000.0])
        if status:
            print(f"Audio status: {status}")
        roll, pitch, yaw = self.imu.get_euler()
        target_yaw = -self.yaw_gain * yaw
        target_pitch = self.pitch_gain * pitch
        alpha = 0.3
        self.filtered_yaw = (1.0 - alpha) * self.filtered_yaw + alpha * target_yaw
        self.filtered_pitch = (1.0 - alpha) * self.filtered_pitch + alpha * target_pitch
        with self._vision_lock:
            active_tracks = [(tid, ss) for tid, ss in self.source_states.items() 
                           if ss.active and ((t_now - ss.last_update_t) <= self.vision_timeout_s)]
        for track_id, ss in active_tracks:
            source_idx = self._allocate_audio_source(track_id)
            if track_id not in self._src_az_ema:
                self._src_az_ema[track_id] = ss.azimuth_deg - yaw
                self._src_el_ema[track_id] = ss.elevation_deg - pitch
                self._src_dist_ema[track_id] = ss.distance_est
                self._src_gain_ema[track_id] = ss.gain
            # CONTINUATION OF phase4_complete.py - Add this to the end of Part 1

    # ... continuing audio_callback method:
            meas_az = self._wrap_deg(ss.azimuth_deg - float(self.filtered_yaw))
            meas_el = float(np.clip(ss.elevation_deg - float(self.filtered_pitch), -90.0, 90.0))
            meas_dist = float(ss.distance_est)
            meas_gain = float(ss.gain)
            
            b_az = float(VISION_CONFIG.get("smooth_beta_az", 0.20))
            b_el = float(VISION_CONFIG.get("smooth_beta_el", 0.20))
            b_d = float(VISION_CONFIG.get("smooth_beta_dist", 0.25))
            b_g = float(VISION_CONFIG.get("gain_smooth_beta", 0.20))
            
            self._src_az_ema[track_id] = self._ema_angle(self._src_az_ema[track_id], meas_az, b_az)
            self._src_el_ema[track_id] = (1.0 - b_el) * self._src_el_ema[track_id] + b_el * meas_el
            self._src_dist_ema[track_id] = (1.0 - b_d) * self._src_dist_ema[track_id] + b_d * meas_dist
            self._src_gain_ema[track_id] = (1.0 - b_g) * self._src_gain_ema[track_id] + b_g * meas_gain
            
            self.sources[source_idx].set_target_position(
                azimuth=self._src_az_ema[track_id],
                elevation=self._src_el_ema[track_id],
                distance=self._src_dist_ema[track_id]
            )
            self.sources[source_idx].set_target_gain(self._src_gain_ema[track_id])
        
        # Fade out inactive tracks
        for track_id in list(self._src_gain_ema.keys()):
            if track_id not in [t[0] for t in active_tracks]:
                b_g = float(VISION_CONFIG.get("gain_smooth_beta", 0.20))
                self._src_gain_ema[track_id] = (1.0 - b_g) * self._src_gain_ema[track_id]
                if track_id in self.track_to_source_idx and self._src_gain_ema[track_id] < 0.01:
                    source_idx = self.track_to_source_idx[track_id]
                    self.sources[source_idx].set_target_gain(0.0)
        
        mixed_output = np.zeros((frames, 2), dtype=np.float32)
        for i, source in enumerate(self.sources):
            if source.is_active:
                chunk = source.get_audio_chunk(frames)
                spatialized = self.spatialize_audio_block(chunk, i)
                mixed_output += spatialized
        
        # Smoothed limiter
        if not hasattr(self, '_limiter_gain'):
            self._limiter_gain = 1.0
        limiter_threshold = 0.7
        limiter_attack = 0.15
        limiter_release = 0.995
        max_val = float(np.max(np.abs(mixed_output)))
        if max_val > limiter_threshold:
            target_gain = limiter_threshold / (max_val + 1e-8)
            self._limiter_gain = min(self._limiter_gain, limiter_attack * target_gain + (1 - limiter_attack) * self._limiter_gain)
        else:
            self._limiter_gain = limiter_release * self._limiter_gain + (1 - limiter_release) * 1.0
        mixed_output *= self._limiter_gain
        
        if self.is_recording:
            self.recorded_frames.append(mixed_output.copy())
            self.recording_duration += frames / self.sample_rate
        outdata[:] = mixed_output

    def start_playback(self):
        if self.is_playing:
            print("Already playing")
            return
        self.is_playing = True
        for source in self.sources:
            source.reset_position()
        self.overlap_buffers = [np.zeros((self.hrir_length - 1, 2), dtype=np.float32) for _ in self.sources]
        print("\n" + "=" * 60)
        print("PHASE 4: MULTI-OBJECT + AMPLITUDE MODULATION")
        print("=" * 60)
        print("\nFeatures:")
        print("  • Multi-object tracking (up to 3 objects)")
        print("  • Amplitude modulation (parking sensor)")
        print("  • World-locked spatial audio")
        print("  • Press Ctrl+C to stop\n")
        self.stream = sd.OutputStream(samplerate=self.sample_rate, channels=2, 
                                     callback=self.audio_callback, blocksize=self.buffer_size)
        self.stream.start()

    def stop_playback(self):
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
        try:
            self.latency_log.close()
        except Exception:
            pass
        self.is_playing = False
        if self.is_recording:
            self.stop_recording()
        print("\nPlayback stopped")

    def start_recording(self):
        if not self.is_playing:
            print("Cannot record - playback not active")
            return
        self.is_recording = True
        self.recorded_frames = []
        self.recording_duration = 0.0
        print("\n🔴 RECORDING STARTED")

    def stop_recording(self, filename="spatial_audio_phase4.wav"):
        if not self.is_recording:
            return
        self.is_recording = False
        if len(self.recorded_frames) == 0:
            print("No audio recorded")
            return
        full_audio = np.vstack(self.recorded_frames)
        sf.write(filename, full_audio, self.sample_rate)
        print(f"\n✅ Recording saved: {filename}")
        print(f"   Duration: {self.recording_duration:.2f}s")
        self.recorded_frames = []
        self.recording_duration = 0.0

    def toggle_recording(self):
        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording()


if __name__ == "__main__":
    print("=" * 70)
    print("PHASE 4: MULTI-OBJECT TRACKING + AMPLITUDE MODULATION")
    print("=" * 70)
    
    try:
        audio_files = ["rain.mp3"]
        
        processor = MultiSourceHRTFAudio(
            audio_files=audio_files,
            sofa_file="MIT_KEMAR_normal_pinna.sofa",
            sample_rate=44100,
            imu_port=5005
        )
        
        processor.start_playback()
        processor.use_world_lock_for_vision = True
        
        vision_thread = VisionYOLOMultiObject(processor)
        vision_thread.start()
        
        print("\n[INFO] System running. Press Ctrl+C to stop.")
        print("[INFO] Press 'R' to toggle recording (if keyboard library installed)")
        
        try:
            import keyboard
            print("[INFO] Keyboard control enabled")
            while True:
                if keyboard.is_pressed('r'):
                    processor.toggle_recording()
                    while keyboard.is_pressed('r'):
                        time.sleep(0.1)
                time.sleep(0.1)
        except ImportError:
            print("[WARN] 'keyboard' library not installed. Recording via 'R' disabled.")
            try:
                while True:
                    time.sleep(0.5)
            except KeyboardInterrupt:
                print("\n[MAIN] Stopping...")
        except KeyboardInterrupt:
            print("\n[MAIN] Stopping...")
        finally:
            try:
                vision_thread.stop()
                vision_thread.join(timeout=2.0)
            except Exception:
                pass
            processor.stop_playback()
    
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nMake sure you have:")
        print("  - rain.wav (audio file)")
        print("  - MIT_KEMAR_normal_pinna.sofa (HRTF data)")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
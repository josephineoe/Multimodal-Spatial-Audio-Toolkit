# Real-Time Spatial Perception Stack

---

## Project Author & Roles

This project was designed, implemented, and documented end-to-end by a single contributor (josephineoe), whose responsibilities span multiple industry-standard roles:

- **Principal Investigator / Project Lead:** Overall vision, architecture, and direction.
- **Software Engineer (Full Stack):** Designed and implemented all core modules (audio, vision, IMU, timing, orchestration).
- **Machine Learning Engineer:** Integrated and configured YOLO object detection for real-time vision.
- **Embedded Systems Engineer:** Managed IMU data acquisition and real-time sensor fusion.
- **Audio DSP Engineer:** Developed HRTF spatialization, SOFA file handling, and real-time audio processing.
- **DevOps / Build Engineer:** Set up dependencies, runtime scripts, and ensured cross-platform operability.
- **Technical Writer:** Authored all documentation, including README, configuration, and usage instructions.
- **QA / Test Engineer:** Validated module functionality, debugged, and ensured system reliability.

This makes the author the equivalent of a "Technical Founder" or "Lead Systems Architect" in industry terms, responsible for end-to-end design, implementation, and delivery.

---

### IMU-Driven HRTF Audio with Camera-Based Object Detection

A real-time perception system that maps the physical environment into 3D spatial audio. An IMU streams head orientation as quaternions over UDP; a camera runs YOLO inference to detect and localize objects; the two streams are fused in real time and used to drive HRTF-spatialized audio, so each detected object produces a binaural sound anchored to its position in the world.

Target deployment: **NVIDIA Jetson** (edge, low-latency).

---

## Architecture

```
┌─────────────────┐     UDP/quaternion     ┌──────────────────────────┐
│  IMU (hardware) │ ──────────────────────▶│  HeadTrackingReceiver    │
│  (e.g. BNO085)  │                        │  quaternion → roll/pitch/ │
└─────────────────┘                        │  yaw  (aerospace Z-Y-X)  │
                                           └────────────┬─────────────┘
                                                        │ head pose
┌─────────────────┐     OpenCV frames      ┌────────────▼─────────────┐
│  Camera         │ ──────────────────────▶│  ObjectDetectionYOLO     │
│  (USB / V4L2 /  │                        │  YOLOv11n @ 8 Hz         │
│   GStreamer)    │                        │  px → azimuth/elevation  │
└─────────────────┘                        │  bbox-height → distance  │
                                           └────────────┬─────────────┘
                                                        │ world-frame target
                                           ┌────────────▼─────────────┐
                                           │  SpatialAudioProcessor   │
                                           │  sensor fusion +         │
                                           │  EMA smoothing           │
                                           │  SOFA HRTF lookup +      │
                                           │  OLA convolution         │
                                           └────────────┬─────────────┘
                                                        │ binaural PCM
                                           ┌────────────▼─────────────┐
                                           │  sounddevice OutputStream│
                                           │  44.1 kHz stereo output  │
                                           └──────────────────────────┘
```

---

## Key Implementation Details

### Sensor Fusion (`hrtf.py → SpatialAudioProcessor.update_vision_target`)
Camera detections arrive in the camera frame (azimuth relative to lens center). Head yaw from the IMU is added to rotate them into world coordinates, keeping each sound source stationary in the room even as the user's head moves:

```python
az_world = az_camera + (YAW_SIGN * yaw_imu)
el_world = el_camera + pitch_imu
```

Both streams write asynchronously; a confidence gate (`conf ≥ 0.25`) prevents low-quality detections from updating the audio position.

### HRTF Spatialization (`hrtf.py`)
- Loads MIT KEMAR SOFA HRTF measurements (nearest-neighbor lookup over azimuth/elevation grid)
- Per-source **overlap-add (OLA)** convolution runs inside the `sounddevice` audio callback — no blocking I/O on the audio thread
- HRIR coefficients are cross-faded between updates (`α = 0.92`) to suppress switching artifacts
- Spatial position per source interpolates with a 50 ms time constant so fast IMU/vision updates don't create audible pops

### Distance Estimation (`vision.py → ObjectDetectionYOLO._estimate_distance_m`)
No depth camera or AI depth model required. Distance is estimated from bounding-box height combined with a per-class prior on real-world object height (e.g. person = 1.7 m, cup = 0.12 m) and the derived vertical focal length:

```
f_px = (frame_h / 2) / tan(VFOV / 2)
dist = (real_height_m × f_px) / bbox_height_px
```

Distance feeds a gain curve that attenuates audio with distance (`gain ∝ ref_dist / dist`).

### Low-Latency Design
| Concern | Solution |
|---|---|
| IMU jitter | Non-blocking UDP socket; quaternion parsed on dedicated daemon thread |
| Vision CPU cost | YOLO runs at a configurable rate (default 8 Hz); camera capture runs independently at 30 fps |
| Audio underruns | OLA convolution is entirely NumPy; buffer size 4096 samples (~93 ms at 44.1 kHz) |
| Position jitter | EMA smoothing on azimuth, elevation, distance, and gain independently |
| Debug overhead | CSV logging (IMU + vision latency) only enabled when recording is active |

---

## Modules

| File         | Responsibility |
|--------------|-----------------------------------------------------------------------------------------------|
| `main.py`    | Orchestrator: CLI, coordinates audio (HRTF), vision, and head-tracking subsystems.            |
| `hrtf.py`    | `SpatialAudioProcessor` — HRTF spatialization, SOFA loading, OLA convolution, sensor fusion;  |
|              | `SpatialAudioSource` — per-source audio state; `DebugLogger` — CSV logging.                   |
| `vision.py`  | `ObjectDetectionYOLO` — camera capture, YOLOv11n inference, distance estimation;              |
|              | `VISION_CONFIG` — all tunable parameters.                                                     |
| `imu.py`     | `HeadTrackingReceiver` — UDP IMU listener, quaternion parsing, Euler angle conversion.        |
| `timing.py`  | `SystemClock` — unified timing reference for all modules, latency measurement utilities.      |

---

## Configuration (`VISION_CONFIG`)

```python
"model_path":    "yolo11n.pt"      # swap for yolo11s/m for accuracy vs speed trade-off
"infer_hz":      8.0               # inference rate — reduce on slower hardware
"hfov_deg":      70.0              # camera horizontal FOV (calibrate per lens)
"target_mode":   "allowed_objects" # or "person_only"
"allowed_classes": {"person", "chair", "cup", ...}
"enable_distance_attenuation": True
"camera_source": "default"         # "usb" for /dev/video0, "gstreamer" for Jetson CSI
```

---

## Dependencies

```
numpy scipy soundfile sounddevice
ultralytics        # YOLOv11
opencv-python      # camera capture + annotation
python-sofa        # SOFA HRTF file loading
```

Required assets:
- `MIT_KEMAR_normal_pinna.sofa` — HRTF measurements
- `rain.wav`, `drums.wav` — spatial audio sources (any mono/stereo WAV)
- `yolo11n.pt` — YOLOv11 nano weights

---

## Running

```bash
# Real-time mode (head-tracked + vision)
python main.py
# → choose 1

# Offline render to WAV (no camera required)
python main.py
# → choose 2, enter duration

# Test modules independently
python hrtf.py    # validates SOFA loading, IMU socket, audio sources
python vision.py  # validates camera access and YOLO model
```

**Runtime commands:**
- `r` — start/stop recording spatial audio output + CSV debug logs
- `d` — toggle live camera feed with bounding-box overlay
- `v` — start/stop vision thread
- `q` — quit

---

## IMU Protocol

The receiver accepts two UDP packet formats on port 5005:

```
# CSV (preferred — includes send timestamp for latency logging)
t_send,qw,qx,qy,qz

# Labeled text
qw: 0.919 qx: 0.057 qy: 0.275 qz: 0.275
```

Quaternions are normalized on receipt and converted to roll/pitch/yaw using standard aerospace (Z-Y-X) convention.

---

## Debug Logging

Recording (`r`) enables per-frame CSV logs in `./debug_logs/`:

| File | Columns |
|---|---|
| `vision_log.csv` | timestamp, az_cam, el_cam, dist_m, conf, class, latency_ms |
| `imu_log.csv` | timestamp, qw/qx/qy/qz, roll, pitch, yaw, latency_ms |

Latency is measured end-to-end from sender timestamp to processing time.

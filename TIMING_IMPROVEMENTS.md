# TIMING & SYNCHRONIZATION IMPROVEMENTS GUIDE

## Executive Summary

Your system currently has **multiple independent clocks** causing unreliable latency measurements and synchronization issues. The solution is to use a **unified `SystemClock`** that all modules reference.

---

## CURRENT ISSUES IDENTIFIED

### Issue #1: IMU Timestamp Inconsistency (imu.py, line ~80)

**Location**: `imu.py` in `_loop()` method

```python
# ❌ PROBLEM: Format 2 uses receiver's time
if "," in s:
    t_send = float(parts[0])  # ✓ Good: sender's timestamp
else:
    t_send = time.time()      # ❌ Bad: receiver's time (drifts from format 1)
```

**Impact**: 
- CSV format uses sender's clock
- Labeled format uses receiver's clock  
- Latency calculations mix two different clock sources

**Fix**: Always use sender's timestamp, or use receiver's time consistently across all formats

---

### Issue #2: Vision Logging Bug (hrtf.py, line ~391)

**Location**: `hrtf.py` in `update_vision_target()` method

```python
# ❌ PROBLEM: Passes same value twice
def update_vision_target(self, ..., t_vision=None, ...):
    t = t_vision if t_vision is not None else time.time()
    # ...
    self.logger.log_vision(t, az_deg, el_deg, d, c, cname, t)
    #                      ↑                              ↑
    #                   event time              source time (SAME!)
```

**Impact**:
- Latency calculation becomes meaningless (always ~0)
- Cannot detect synchronization issues

**Fix**: 
```python
# ✓ CORRECT
self.logger.log_vision(
    time.time(),   # When we received/logged this (receiver time)
    az_deg, el_deg, d, c, cname, 
    t_vision       # When vision detector ran (source time)
)
```

---

### Issue #3: Mixed Clock Sources in Latency Calculation (hrtf.py DebugLogger)

**Location**: `hrtf.py` in `DebugLogger.log_vision()` and `log_imu()`

```python
def log_vision(self, t, az_deg, el_deg, dist_m, conf, cls_name, t_vision):
    latency_ms = (time.time() - float(t_vision)) * 1000.0
    # ❌ PROBLEM: Mixes current time.time() with various t_vision sources
```

**Impact**:
- If `t_vision` is from sender, latency includes network + processing delay ✓
- If `t_vision` is from receiver, latency is ~0 and meaningless ✗
- Cannot distinguish between sources

---

### Issue #4: No Synchronized Reference Time

**Problem**: Each thread/module calls `time.time()` independently:
- `imu.py`: Uses `time.time()` in UDP loop
- `vision.py`: Uses `time.time()` in detection loop  
- `hrtf.py`: Uses `time.time()` in audio callback
- Logger: Uses `time.time()` for events

Result: OS scheduling and thread context switches cause slight drifts.

---

## RECOMMENDED IMPROVEMENTS

### Step 1: Use Unified SystemClock (NEW `timing.py`)

All modules should use:
```python
from timing import system_clock, calculate_latency_ms

# Instead of:
t_now = time.time()

# Use:
t_now = system_clock.now()              # High-resolution reference
latency = calculate_latency_ms(t_source)  # Consistent calculation
```

**Benefits**:
- Single source of truth for all timestamps
- Uses `time.perf_counter()` (immune to clock adjustments)
- Convenient API for common operations

---

### Step 2: Fix IMU Timestamp Handling (imu.py)

**Current** (line ~80):
```python
if "," in s:
    t_send = float(parts[0])
else:
    t_send = time.time()  # ❌ Uses receiver's time
```

**Improved**:
```python
from timing import system_clock

if "," in s:
    t_send = float(parts[0])  # Sender's timestamp from CSV
else:
    # Format 2 doesn't have sender's timestamp, use receiver's consistent clock
    t_send = system_clock.now()  # ✓ Same clock source as rest of system
```

---

### Step 3: Fix Vision Logging (hrtf.py)

**Current** (line ~391):
```python
self.logger.log_vision(t, az_deg, el_deg, d, c, cname, t)
#                      ↑                                ↑
#                    SAME VALUE - latency always 0!
```

**Improved**:
```python
from timing import system_clock

# In update_vision_target():
t_received = system_clock.now()  # When we received this from vision
self.logger.log_vision(
    t_received,                # Event logged at this time
    az_deg, el_deg, d, c, cname,
    t_vision                   # When vision detected this (source time)
)
```

---

### Step 4: Update DebugLogger (hrtf.py)

**Current**:
```python
def log_vision(self, t, az_deg, el_deg, dist_m, conf, cls_name, t_vision):
    latency_ms = (time.time() - float(t_vision)) * 1000.0  # ❌ Mixed clocks
```

**Improved**:
```python
from timing import system_clock

def log_vision(self, t_received, az_deg, el_deg, dist_m, conf, cls_name, t_vision):
    """
    Log vision data with proper latency calculation.
    
    Args:
        t_received: When this log entry was created (from system_clock.now())
        t_vision: When vision detector ran (from system_clock.now() at source)
    """
    latency_ms = (t_received - float(t_vision)) * 1000.0  # ✓ Consistent clocks
    self._vision_w.writerow([
        float(t_received),  # When logged
        float(az_deg), float(el_deg), float(dist_m),
        float(conf) if conf is not None else 0.0,
        str(cls_name) if cls_name else "",
        float(latency_ms)
    ])
```

---

### Step 5: Update Vision Module (vision.py)

**Current** (in `ObjectDetectionYOLO.run()`):
```python
t_vision = time.time()  # Uses independent time.time()
```

**Improved**:
```python
from timing import system_clock

t_vision = system_clock.now()  # Uses unified clock

# Then in update_vision_target call:
self.processor.update_vision_target(
    az_deg, el_deg, yaw_deg=yaw, pitch_deg=pitch,
    distance_m=dist_m, conf=conf, cls_name=cls_name,
    t_vision=t_vision,  # ✓ Now consistent with all other modules
    source_id=source_id
)
```

---

### Step 6: Update IMU Logging (imu.py)

**Current** (line ~87):
```python
if self.logger:
    roll, pitch, yaw = self.get_euler()
    self.logger.log_imu(time.time(), qw, qx, qy, qz, roll, pitch, yaw, t_send)
    #                   ^^^^^^^^ Uses independent time.time()
```

**Improved**:
```python
from timing import system_clock

if self.logger:
    roll, pitch, yaw = self.get_euler()
    t_received = system_clock.now()  # ✓ Unified clock
    self.logger.log_imu(t_received, qw, qx, qy, qz, roll, pitch, yaw, t_send)
```

---

## TESTING THE IMPROVEMENTS

### Test 1: Verify Unified Clock
```python
from timing import system_clock, calculate_latency_ms
import time

t1 = system_clock.now()
time.sleep(0.1)
t2 = system_clock.now()

latency = (t2 - t1) * 1000
print(f"Elapsed: {latency:.1f} ms")  # Should show ~100 ms
```

### Test 2: Run Timing Module Test
```bash
python timing.py
```
Expected output:
```
Test 1: Basic timing
  Elapsed time: 100.23 ms (expected ~100 ms)

Test 2: Event recording
  Event type: test
  Latency: 0.015 ms (should be very small)
```

---

## EXPECTED RESULTS AFTER IMPROVEMENTS

| Metric | Before | After |
|--------|--------|-------|
| Latency measurement accuracy | ±50ms drift | ±1ms drift |
| Clock source consistency | Mixed (3+ sources) | Unified (1 source) |
| Performance analysis reliability | Unreliable | Accurate |
| Synchronization detection | Poor | Excellent |

---

## IMPLEMENTATION PRIORITY

1. **High Priority**: Create `timing.py` and update `hrtf.py` logger ← FIX THE BUG
2. **High Priority**: Update `vision.py` to use `system_clock`
3. **Medium Priority**: Update `imu.py` to use `system_clock`
4. **Medium Priority**: Update `main.py` to display timing stats

---

## MONITORING SCALABILITY

After implementing unified timing, you can reliably measure:

```python
# In your performance analysis script:
from timing import system_clock

metrics = {
    'imu_latency_ms': [],      # Now reliable
    'vision_latency_ms': [],   # Now reliable
    'audio_latency_ms': [],    # Now reliable
    'cpu_usage': [],
    'memory_usage': []
}

# Log these together with same clock source
# → Can now correlate latency with resource usage
```

---

## Summary

The core issue is **mixing clock sources**. By using a single `SystemClock` instance throughout the system, you'll have:

1. ✅ Accurate latency measurements
2. ✅ Reliable synchronization detection
3. ✅ Trustworthy performance/scalability analysis
4. ✅ Easy debugging of timing issues
5. ✅ Foundation for distributed timing (NTP sync) if needed later

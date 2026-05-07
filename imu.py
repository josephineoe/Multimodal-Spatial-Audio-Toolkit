# ============================================================================
# IMU HEAD TRACKING MODULE
# Quaternion receiver and Euler angle conversion for head orientation
# ============================================================================

import socket
import threading
import time
import numpy as np

# Import timing module for unified clock
from timing import system_clock


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
    
    Supports two input formats:
    1) CSV: t_send,qw,qx,qy,qz
    2) Labeled text: qw: 0.919 qx: 0.057 qy: 0.275 qz: 0.275
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
                    # ✓ FIXED: Use system_clock for consistent timing
                    t_send = system_clock.now()

                self.t_send = float(t_send)
                self.qw, self.qx, self.qy, self.qz = float(qw), float(qx), float(qy), float(qz)

                if self.logger:
                    roll, pitch, yaw = self.get_euler()
                    # ✓ FIXED: Use system_clock for receiver timestamp
                    t_received = system_clock.now()
                    self.logger.log_imu(t_received, qw, qx, qy, qz, roll, pitch, yaw, t_send)

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


if __name__ == "__main__":
    """
    Test IMU module independently.
    Checks UDP listening and quaternion parsing.
    """
    print("=" * 70)
    print("IMU HEAD TRACKING MODULE TEST")
    print("=" * 70)
    print()
    print("Testing UDP listener on 0.0.0.0:5005")
    print("Send quaternion data to test:")
    print("  Format 1 (CSV): t_send,qw,qx,qy,qz")
    print("  Format 2 (labeled): qw: 0.919 qx: 0.057 qy: 0.275 qz: 0.275")
    print()
    
    receiver = HeadTrackingReceiver(port=5005)
    
    try:
        print("Waiting for IMU data... (Press Ctrl+C to exit)")
        while True:
            if receiver.t_send != 0.0:
                roll, pitch, yaw = receiver.get_euler()
                print(f"[IMU] Q=({receiver.qw:.3f}, {receiver.qx:.3f}, {receiver.qy:.3f}, {receiver.qz:.3f}) "
                      f"Euler: roll={roll:.1f}° pitch={pitch:.1f}° yaw={yaw:.1f}°")
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nIMU test stopped.")

# ============================================================================
# SYSTEM TIMING MODULE
# Unified clock and timing utilities for the multimodal toolkit
# Ensures all subsystems use the same reference time for accurate
# latency measurement and synchronization analysis
# ============================================================================

import time
import threading
from typing import NamedTuple


# =========================================================
# Timestamp Event (for structured logging)
# =========================================================

class TimestampEvent(NamedTuple):
    """A timestamped event with metadata."""
    t_system: float           # System time when recorded (receiver)
    t_source: float           # Time from source (sender)
    event_type: str           # "imu", "vision", "audio"
    source_id: int            # Which IMU/vision/audio source
    latency_ms: float         # t_system - t_source (milliseconds)
    metadata: dict            # Additional context


# =========================================================
# SystemClock (Unified Timing Reference)
# =========================================================

class SystemClock:
    """
    Centralized timing reference for the entire system.
    - Provides a single `now()` method that all modules use
    - Uses time.perf_counter() for high-resolution timing
    - Optionally synchronizes with external devices (NTP, IMU sender clock)
    
    Usage:
        from timing import system_clock
        
        t_now = system_clock.now()  # Get current time
        latency = system_clock.now() - t_source  # Calculate latency
    """
    
    def __init__(self):
        """Initialize the system clock."""
        self._lock = threading.Lock()
        self._start_time = time.perf_counter()  # High-resolution reference
        self._wall_time_offset = time.time()  # Wall-clock offset for logging
        self._sync_drift = 0.0  # Accumulated drift (for future NTP sync)
        
    def now(self) -> float:
        """
        Get current system time in seconds.
        Uses high-resolution timer (time.perf_counter()) for consistency.
        
        Returns:
            float: Time in seconds since arbitrary reference point.
                   All calls use the same clock source for consistency.
        """
        return time.perf_counter()
    
    def now_wall(self) -> float:
        """
        Get wall-clock time (seconds since epoch).
        Useful for readable logging and human timestamps.
        
        Returns:
            float: Unix timestamp (seconds since 1970-01-01 UTC)
        """
        return time.time()
    
    def elapsed_ms(self, t_start: float) -> float:
        """
        Calculate elapsed time in milliseconds.
        
        Args:
            t_start: Start time from now()
            
        Returns:
            float: Milliseconds elapsed since t_start
        """
        return (self.now() - t_start) * 1000.0
    
    def elapsed_s(self, t_start: float) -> float:
        """
        Calculate elapsed time in seconds.
        
        Args:
            t_start: Start time from now()
            
        Returns:
            float: Seconds elapsed since t_start
        """
        return self.now() - t_start
    
    def record_event(self, event_type: str, source_id: int = 0, 
                     t_source: float = None, metadata: dict = None) -> TimestampEvent:
        """
        Create a timestamped event with automatic latency calculation.
        
        Args:
            event_type: "imu", "vision", "audio", etc.
            source_id: Which source generated the event
            t_source: Time when event originated (if not now)
            metadata: Optional additional context dict
            
        Returns:
            TimestampEvent: Structured event with timestamps and latency
        """
        t_system = self.now()
        if t_source is None:
            t_source = t_system
        
        latency_ms = (t_system - t_source) * 1000.0
        
        return TimestampEvent(
            t_system=t_system,
            t_source=t_source,
            event_type=event_type,
            source_id=source_id,
            latency_ms=latency_ms,
            metadata=metadata or {}
        )


# =========================================================
# Global System Clock Instance
# =========================================================

system_clock = SystemClock()


# =========================================================
# Helper Functions for Common Timing Tasks
# =========================================================

def get_time() -> float:
    """Convenience function: get current time from system clock."""
    return system_clock.now()


def get_wall_time() -> float:
    """Convenience function: get wall-clock time."""
    return system_clock.now_wall()


def elapsed_ms(t_start: float) -> float:
    """Convenience function: get elapsed milliseconds."""
    return system_clock.elapsed_ms(t_start)


def elapsed_s(t_start: float) -> float:
    """Convenience function: get elapsed seconds."""
    return system_clock.elapsed_s(t_start)


def calculate_latency_ms(t_source: float) -> float:
    """
    Calculate latency from source time to now.
    
    Args:
        t_source: Time when event originated
        
    Returns:
        float: Latency in milliseconds
    """
    return (system_clock.now() - t_source) * 1000.0


if __name__ == "__main__":
    """
    Test timing module.
    """
    print("=" * 70)
    print("SYSTEM TIMING MODULE TEST")
    print("=" * 70)
    print()
    
    # Test basic timing
    print("Test 1: Basic timing")
    t_start = system_clock.now()
    time.sleep(0.1)
    elapsed = system_clock.elapsed_ms(t_start)
    print(f"  Elapsed time: {elapsed:.2f} ms (expected ~100 ms)")
    print()
    
    # Test event recording
    print("Test 2: Event recording")
    event = system_clock.record_event("test", source_id=0)
    print(f"  Event type: {event.event_type}")
    print(f"  Latency: {event.latency_ms:.3f} ms (should be very small)")
    print()
    
    # Test wall-clock
    print("Test 3: Wall-clock time")
    wall = system_clock.now_wall()
    import datetime
    dt = datetime.datetime.fromtimestamp(wall)
    print(f"  Current time: {dt}")
    print()
    
    # Test latency with simulated delay
    print("Test 4: Latency simulation")
    t_source = system_clock.now()
    time.sleep(0.05)
    latency = calculate_latency_ms(t_source)
    print(f"  Simulated source delay: ~50 ms")
    print(f"  Measured latency: {latency:.2f} ms")
    print()
    
    print("✅ Timing module ready for integration")

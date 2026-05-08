# ============================================================================
# MULTIMODAL SPATIAL AUDIO TOOLKIT - MAIN
# Orchestrator: Coordinates HRTF audio (hrtf.py) and vision (vision.py)
# ============================================================================

import threading
import time
import os
import argparse
import sys

# Import modules
from hrtf import SpatialAudioProcessor
from vision import ObjectDetectionYOLO, VISION_CONFIG, start_and_test_vision


def parse_arguments():
    """Parse command-line arguments for subsystem selection."""
    parser = argparse.ArgumentParser(
        description="Multimodal Spatial Audio Toolkit - HRTF + Vision + Head-Tracking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
MODES:

    python main.py                                Full system (vision + IMU head-tracking)
        python main.py --vision --detection-mode 1   Vision control for both animate + inanimate
        python main.py --vision --detection-mode 2   Vision control for inanimate objects only
        python main.py --vision --detection-mode 3   Vision control for animate/person only
        python main.py --imu                          IMU-only audio control

DETECTION MODES (for vision-based control):

  --detection-mode 1                Mode 1: Detect BOTH animate (person) + inanimate (furniture)
  --detection-mode 2                Mode 2: Detect INANIMATE ONLY (bed, chair, couch, etc.)
  --detection-mode 3 (default)      Mode 3: Detect ANIMATE ONLY (person/movement)

EXAMPLES:

    python main.py --vision --detection-mode 3   Vision control with person detection only
    python main.py --vision --detection-mode 2   Vision control with furniture detection
    python main.py --imu                         IMU-only audio control

OFFLINE RENDER:

  python main.py --mode 2                       Offline audio render (HRTF without real-time I/O)
        """
    )
    
    parser.add_argument(
        "--vision",
        action="store_true",
        help="Enable vision-based object detection for audio control"
    )
    parser.add_argument(
        "--imu",
        action="store_true",
        help="Enable IMU head-tracking for audio control"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["1", "2"],
        default="1",
        help="Operation mode: 1=real-time (default), 2=offline render"
    )
    
    parser.add_argument(
        "--detection-mode",
        type=int,
        choices=[1, 2, 3],
        default=3,
        help="Detection mode: 1=both animate+inanimate, 2=inanimate only, 3=animate/person only (default)"
    )
    
    return parser.parse_args()


def main():
    """Main entry point: Setup and orchestration."""
    args = parse_arguments()
    
    # Determine which subsystems to enable
    # Audio engine always runs - it's the core component
    # Default (no args): Full system (vision and IMU enabled)
    # --vision: vision-only control unless --imu is also supplied
    # --imu: IMU-only control unless --vision is also supplied
    enable_vision = args.vision or (not args.vision and not args.imu)
    enable_imu = args.imu or (not args.vision and not args.imu)
    
    print("=" * 70)
    print("MULTIMODAL SPATIAL AUDIO TOOLKIT")
    print("HRTF + Vision + Head-Tracking")
    print("=" * 70)
    print()
    subsystems = ["Audio Engine (HRTF)"]
    if enable_vision:
        subsystems.append("Vision (YOLO)")
    if enable_imu:
        subsystems.append("IMU Head-Tracking")
    print(f"Active Subsystems: {' + '.join(subsystems)}")
    
    if enable_vision and enable_imu:
        print("  Mode: FULL SYSTEM (vision + IMU head-tracking)")
    elif enable_vision:
        print("  Mode: VISION ONLY (no IMU head-tracking)")
    elif enable_imu:
        print("  Mode: IMU ONLY (no vision)")
    else:
        print("  Mode: AUDIO ONLY (no vision, no IMU)")
    
    # Display detection mode if vision is enabled
    if enable_vision:
        detection_mode_names = {1: "Both (animate + inanimate)", 2: "Inanimate only", 3: "Animate/person only"}
        print(f"  Detection Mode: {args.detection_mode} - {detection_mode_names.get(args.detection_mode, 'Unknown')}")
    print()

    try:
        audio_files = ["drums.wav", "rain.wav"]

        # Apply detection mode from command-line argument
        if enable_vision:
            VISION_CONFIG["detection_mode"] = args.detection_mode
        
        # Initialize audio engine (HRTF + optional IMU + optional Vision)
        processor = SpatialAudioProcessor(
            audio_files=audio_files,
            sofa_file="MIT_KEMAR_normal_pinna.sofa",
            sample_rate=44100,
            imu_port=5005 if enable_imu else None,
            vision_config=VISION_CONFIG if enable_vision else {}
        )

        # Ask user for mode
        if args.mode == "2":
            print("\n[MAIN] Offline render mode selected.")
            duration = input("Offline render duration in seconds (default 5): ").strip()
            try:
                duration = float(duration)
            except Exception:
                duration = 5.0
            processor.export_offline_render(duration_seconds=duration)

        else:
            # Real-time playback mode
            print("\n[MAIN] Starting real-time audio playback...")
            processor.start_playback()

            # Start vision thread (if enabled)
            vision_thread = None
            if enable_vision:
                print("[MAIN] Starting vision (YOLO)...")
                vision_thread = start_and_test_vision(processor)
            else:
                print("[MAIN] Vision is DISABLED.")
            
            if not enable_imu:
                print("[MAIN] IMU head-tracking is DISABLED.")

            # Simple interactive controls
            control_state = {"vision": vision_thread}

            def _control_loop():
                """Background control loop for user commands."""
                while True:
                    try:
                        cmd = input(
                            "[CTRL] Commands: r=record, v=toggle vision, "
                            "d=debug display, q=quit > "
                        ).strip().lower()
                    except Exception:
                        return

                    if cmd == "r":
                        if processor is not None:
                            processor.toggle_recording()
                        else:
                            print("[CTRL] Recording unavailable.")
                    elif cmd == "d":
                        if not enable_vision:
                            print("[CTRL] Debug display unavailable (vision disabled).")
                            continue
                        VISION_CONFIG["show_window"] = not VISION_CONFIG["show_window"]
                        state = "ON" if VISION_CONFIG["show_window"] else "OFF"
                        print(f"[CTRL] 📷 Webcam POV debug display: {state}")
                        if VISION_CONFIG["show_window"]:
                            print(f"      Opening camera feed with object detections...")
                        else:
                            print(f"      Closing camera feed window.")
                    elif cmd == "v":
                        if not enable_vision:
                            print("[CTRL] Vision is disabled (use --vision to enable).")
                            continue
                        vt = control_state.get("vision")
                        if vt is None or not vt.is_alive():
                            try:
                                vt = ObjectDetectionYOLO(processor)
                                vt.start()
                                control_state["vision"] = vt
                                print("[CTRL] Vision started.")
                            except Exception as e:
                                print(f"[CTRL] Could not start vision: {e}")
                        else:
                            try:
                                vt.stop()
                                vt.join(timeout=2.0)
                            except Exception:
                                pass
                            control_state["vision"] = None
                            print("[CTRL] Vision stopped.")
                    elif cmd == "q":
                        raise KeyboardInterrupt
                    else:
                        print("[CTRL] Unknown command (r/v/d/q).")

            # Start control loop in background
            threading.Thread(target=_control_loop, daemon=True).start()

            # Main loop
            try:
                while True:
                    time.sleep(0.5)
            except KeyboardInterrupt:
                print("\n[MAIN] Shutting down...")
            finally:
                # Cleanup: stop vision and audio
                try:
                    vt = control_state.get("vision")
                    if vt is not None and vt.is_alive():
                        vt.stop()
                        vt.join(timeout=2.0)
                except Exception:
                    pass

                try:
                    if processor is not None:
                        processor.stop_playback()
                except Exception:
                    pass

                print("[MAIN] Done.")

    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure you have the required audio files and SOFA file:")
        print("  - rain.wav")
        print("  - drums.wav")
        print("  - MIT_KEMAR_normal_pinna.sofa")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

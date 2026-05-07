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
Examples:
  python main.py                        # Run both audio (HRTF) and vision
  python main.py --audio-only           # Run audio (HRTF) only, skip vision
  python main.py --vision-only          # Run vision only, skip audio (HRTF)
  python main.py --mode 2               # Offline render (no vision)
        """
    )
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--audio-only",
        action="store_true",
        help="Run audio (HRTF) processing only, no vision"
    )
    group.add_argument(
        "--vision-only",
        action="store_true",
        help="Run vision only, no audio (HRTF) processing"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["1", "2"],
        default="1",
        help="Operation mode: 1=real-time (default), 2=offline render"
    )
    
    return parser.parse_args()


def main():
    """Main entry point: Setup and orchestration."""
    args = parse_arguments()
    
    # Determine which subsystems to enable
    enable_audio = not args.vision_only
    enable_vision = not args.audio_only
    
    print("=" * 70)
    print("MULTIMODAL SPATIAL AUDIO TOOLKIT")
    print("HRTF + Vision + Head-Tracking")
    print("=" * 70)
    print()
    print(f"Subsystems: {'Audio' if enable_audio else ''}"
          f"{' + ' if enable_audio and enable_vision else ''}"
          f"{'Vision' if enable_vision else ''}")
    print()

    try:
        audio_files = ["drums.wav", "rain.wav"]

        processor = None
        if enable_audio:
            processor = SpatialAudioProcessor(
                audio_files=audio_files,
                sofa_file="MIT_KEMAR_normal_pinna.sofa",
                sample_rate=44100,
                imu_port=5005,
                vision_config=VISION_CONFIG
            )
        else:
            print("[MAIN] Audio (HRTF) processing is DISABLED.")

        # Ask user for mode
        if args.mode == "2":
            if not enable_audio:
                print("[MAIN] Offline render requires audio (HRTF) to be enabled.")
                print("[MAIN] Use: python main.py --mode 2")
                return
            
            duration = input("Offline render duration in seconds (default 5): ").strip()
            try:
                duration = float(duration)
            except Exception:
                duration = 5.0
            processor.export_offline_render(duration_seconds=duration)

        else:
            # Real-time playback mode
            if enable_audio:
                processor.start_playback()
            else:
                print("[MAIN] Real-time playback (audio disabled - vision only).")

            # Start vision thread (if enabled)
            vision_thread = None
            if enable_vision:
                vision_thread = start_and_test_vision(processor)
            else:
                print("[MAIN] Vision processing is DISABLED.")

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
                        if enable_audio and processor is not None:
                            processor.toggle_recording()
                        else:
                            print("[CTRL] Recording unavailable (audio disabled).")
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
                            print("[CTRL] Vision is disabled (use without --vision-only to enable).")
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
                    if enable_audio and processor is not None:
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

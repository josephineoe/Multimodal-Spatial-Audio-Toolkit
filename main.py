# ============================================================================
# MULTIMODAL SPATIAL AUDIO TOOLKIT - MAIN
# Orchestrator: Coordinates HRTF audio (hrtf.py) and vision (vision.py)
# ============================================================================

import threading
import time
import os

# Import modules
from hrtf import SpatialAudioProcessor
from vision import ObjectDetectionYOLO, VISION_CONFIG, start_and_test_vision


def main():
    """Main entry point: Setup and orchestration."""
    print("=" * 70)
    print("MULTIMODAL SPATIAL AUDIO TOOLKIT")
    print("HRTF + Vision + Head-Tracking")
    print("=" * 70)

    try:
        # Audio files
        audio_files = ["rain.wav", "drums.wav"]

        # Initialize audio processor with vision config
        processor = SpatialAudioProcessor(
            audio_files=audio_files,
            sofa_file="MIT_KEMAR_normal_pinna.sofa",
            sample_rate=44100,
            imu_port=5005,
            vision_config=VISION_CONFIG
        )

        # Ask user for mode
        mode = input(
            "Choose mode:\n"
            "  1. Real-time playback (head-tracked + vision)\n"
            "  2. Offline render to WAV (no vision)\n"
            "Choice (1/2): "
        ).strip()

        if mode == "2":
            # Offline render mode
            duration = input("Offline render duration in seconds (default 5): ").strip()
            try:
                duration = float(duration)
            except Exception:
                duration = 5.0
            processor.export_offline_render(duration_seconds=duration)

        else:
            # Real-time playback mode with vision
            processor.start_playback()

            # Start vision thread
            vision_thread = start_and_test_vision(processor)

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
                        processor.toggle_recording()
                    elif cmd == "d":
                        VISION_CONFIG["show_window"] = not VISION_CONFIG["show_window"]
                        state = "ON" if VISION_CONFIG["show_window"] else "OFF"
                        print(f"[CTRL] 📷 Webcam POV debug display: {state}")
                        if VISION_CONFIG["show_window"]:
                            print(f"      Opening camera feed with object detections...")
                        else:
                            print(f"      Closing camera feed window.")
                    elif cmd == "v":
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

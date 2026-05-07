# Multimodal Spatial Audio Toolkit - Changelog

## 2026-05-07 - Dynamic Audio Source Activation (14:30)

### Summary
Fixed critical audio playback behavior: audio sources now activate/deactivate based on object detection rather than always playing all sources.

### Changes Made

#### 1. **Dynamic Source Activation** (hrtf.py - line ~674)
- Added `source.is_active = fresh` in audio_callback's second pass
- Sources now only play audio when actively detected by vision within timeout window
- When an object leaves detection window, audio stops after 0.75s timeout
- **Effect**: Only detected people have audio playing; undetected sources produce silence

#### 2. **Explicit Source Deactivation API** (hrtf.py - new method)
- Added `deactivate_source(source_id)` method for explicit control
- Deactivates both source.is_active and source_state.active
- Useful for manual control or testing individual sources
- **Usage**: `processor.deactivate_source(0)` to silence source 0

#### 3. **Audio Behavior** (User-Facing)
- **0 objects detected** → All audio sources silent (no mixing of empty sources)
- **1 object detected** → Only source 0 (drums.wav) plays, centered on that person
- **2 objects detected** → Source 0 + Source 1 (rain.wav) both play at different spatial positions
- **Object leaves** → That source fades out over 0.75s timeout, then goes silent

### Technical Details
- Detection freshness determined by: `fresh = active && (now - last_update_t <= 0.75s)`
- Timeout is configurable via VISION_CONFIG['no_detection_fade_s']
- Head-tracking (IMU) always active as fallback for spatial audio base
- Gain attenuation still applies based on detection confidence + distance

### Files Modified
- hrtf.py (2 edits: audio_callback source.is_active, deactivate_source method)

### Testing Checklist
- [ ] python main.py → Start audio, move person in view (audio follows)
- [ ] Remove person from view → Audio stops after 0.75s
- [ ] Add second person → Second audio source starts
- [ ] Remove second person → Second audio stops, first continues
- [ ] python main.py --audio-only → Audio with IMU head-tracking only (no vision control)
- [ ] python main.py --vision-only → Audio with vision only (IMU times out)

---

## 2026-05-07 - Agent Changelog Framework Established (14:00)
- Created empty CHANGELOG.md per AGENTS.md requirements
- Framework ready for tracking all future changes with timestamps and file references

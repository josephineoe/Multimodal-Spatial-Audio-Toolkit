# ============================================================================
# MAZE DEMO — Multimodal Spatial Audio Toolkit
# Replaces IMU + vision.py with keyboard controls + 2D raycaster.
# hrtf.py is used UNCHANGED.
#
# Controls: WASD or arrow keys to move/rotate, Q to quit
# Install: pip install pygame numpy soundfile sounddevice scipy
# ============================================================================

import math
import threading
import time

import numpy as np
import pygame

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from hrtf import SpatialAudioProcessor

# ---------------------------------------------------------------------------
# Maze definition  (1 = wall, 0 = open, G = goal cell encoded as 2)
# ---------------------------------------------------------------------------
MAZE = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1],
    [1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1],
    [1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
    [1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 2, 1],  # 2 = goal
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
]

CELL_SIZE  = 56          # pixels per cell in the display
ROWS       = len(MAZE)
COLS       = len(MAZE[0])

# Audio detection radius (cells). Obstacles further away are silenced.
WALL_RADIUS   = 4.0
GOAL_RADIUS   = 8.0

# How many wall sources to track simultaneously (maps to audio sources 0..N-2)
MAX_WALL_SOURCES = 1     # keep 1 so we only need rain.wav; increase if you add more wav files

# Source IDs
GOAL_SOURCE_ID = 1       # drums.wav = goal beacon
WALL_SOURCE_ID = 0       # rain.wav  = nearest wall

# Turn speed (degrees per keypress tick)
TURN_STEP = 10.0
MOVE_STEP = 0.15         # cells per keypress tick

# ---------------------------------------------------------------------------
# Minimal fake IMU that lets us set yaw/pitch directly
# ---------------------------------------------------------------------------
class FakeIMU:
    """Drop-in replacement for HeadTrackingReceiver. No UDP needed."""
    def __init__(self):
        self.yaw_deg   = 0.0    # 0 = North, 90 = East, etc.
        self.pitch_deg = 0.0
        self.t_send    = time.time()

    def get_euler(self):
        return 0.0, self.pitch_deg, self.yaw_deg   # roll, pitch, yaw

    @property
    def qw(self): return 1.0
    @property
    def qx(self): return 0.0
    @property
    def qy(self): return 0.0
    @property
    def qz(self): return 0.0


# ---------------------------------------------------------------------------
# Maze helpers
# ---------------------------------------------------------------------------
def cell_is_wall(row, col):
    if row < 0 or row >= ROWS or col < 0 or col >= COLS:
        return True
    return MAZE[row][col] == 1

def cell_is_goal(row, col):
    if row < 0 or row >= ROWS or col < 0 or col >= COLS:
        return False
    return MAZE[row][col] == 2

def find_goal():
    for r in range(ROWS):
        for c in range(COLS):
            if MAZE[r][c] == 2:
                return (r + 0.5, c + 0.5)   # centre of goal cell
    return None

def find_nearest_wall(player_row, player_col, radius):
    """Return (rel_row, rel_col, dist) for the closest wall within radius, or None."""
    best = None
    best_dist = radius + 1
    pr, pc = player_row, player_col
    ri = int(radius) + 1
    for dr in range(-ri, ri + 1):
        for dc in range(-ri, ri + 1):
            nr, nc = int(pr + dr), int(pc + dc)
            if cell_is_wall(nr, nc):
                dist = math.sqrt(dr * dr + dc * dc)
                if dist < best_dist:
                    best_dist = dist
                    best = (nr + 0.5 - pr, nc + 0.5 - pc, best_dist)
    return best


# ---------------------------------------------------------------------------
# Azimuth calculation: world direction → relative to player heading
# ---------------------------------------------------------------------------
def world_angle_to_azimuth(dx_col, dy_row, player_yaw_deg):
    """
    dx_col: east-positive offset in cells
    dy_row: south-positive offset in cells (screen coords, y grows down)
    player_yaw_deg: 0=north, 90=east

    Returns azimuth_deg relative to player heading:
      0 = straight ahead, 90 = right, -90 = left, 180 = behind
    """
    # World angle: 0=north, 90=east (convert from screen coords)
    world_angle = math.degrees(math.atan2(dx_col, -dy_row))  # north=0, CW positive
    # Relative angle = world angle - player yaw
    rel = (world_angle - player_yaw_deg + 180) % 360 - 180
    return rel


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    pygame.init()
    screen = pygame.display.set_mode((COLS * CELL_SIZE, ROWS * CELL_SIZE))
    pygame.display.set_caption("Spatial Audio Maze — WASD to move, Q to quit")
    clock = pygame.time.Clock()
    font  = pygame.font.SysFont("monospace", 16)

    # Player state (floating point, in cell units)
    player_row = 1.5
    player_col = 1.5
    player_yaw = 0.0   # degrees, 0 = north

    # ---- Audio setup ----
    VISION_CONFIG = {
        "gate_conf_thres":           0.0,   # accept all detections from maze
        "enable_distance_attenuation": True,
        "distance_ref_m":            1.4,
        "no_detection_fade_s":       0.5,
        "gain_min":                  0.0,
        "gain_max":                  1.0,
    }

    print("[MAZE] Initializing audio processor...")
    processor = SpatialAudioProcessor(
        audio_files=["rain.wav", "drums.wav"],
        sofa_file="MIT_KEMAR_normal_pinna.sofa",
        sample_rate=44100,
        imu_port=5005,          # port still bound but never used
        vision_config=VISION_CONFIG,
    )

    # Inject our fake IMU in place of the real HeadTrackingReceiver
    fake_imu = FakeIMU()
    processor.imu = fake_imu

    processor.start_playback()
    print("[MAZE] Audio running. Use WASD / arrows to navigate.\n")

    goal_pos = find_goal()  # (row, col) centre of goal cell

    # ---- Colours ----
    CLR_WALL   = (60, 60, 80)
    CLR_OPEN   = (220, 220, 230)
    CLR_GOAL   = (80, 200, 120)
    CLR_PLAYER = (240, 100, 60)
    CLR_ARROW  = (240, 100, 60)
    CLR_TEXT   = (30, 30, 50)

    running = True
    won     = False

    def audio_update_loop():
        """Background thread: push maze state → SpatialAudioProcessor at 10 Hz."""
        while running:
            pr, pc, yaw = player_row, player_col, player_yaw

            # ---- Wall source ----
            wall = find_nearest_wall(pr, pc, WALL_RADIUS)
            if wall:
                drow, dcol, dist_cells = wall
                az = world_angle_to_azimuth(dcol, drow, yaw)
                dist_m = max(dist_cells * 0.5, 0.2)   # scale: 1 cell ≈ 0.5 m
                processor.update_vision_target(
                    azimuth_deg=az,
                    elevation_deg=0.0,
                    yaw_deg=0.0,       # already baked into az
                    pitch_deg=0.0,
                    distance_m=dist_m,
                    conf=0.9,
                    cls_name="wall",
                    source_id=WALL_SOURCE_ID,
                )
            else:
                # No wall nearby: mute source
                with processor._source_states_lock:
                    processor.source_states[WALL_SOURCE_ID].active = False

            # ---- Goal source ----
            if goal_pos:
                grow, gcol = goal_pos
                drow = grow - pr
                dcol = gcol - pc
                dist_cells = math.sqrt(drow * drow + dcol * dcol)
                if dist_cells <= GOAL_RADIUS:
                    az = world_angle_to_azimuth(dcol, drow, yaw)
                    dist_m = max(dist_cells * 0.5, 0.2)
                    processor.update_vision_target(
                        azimuth_deg=az,
                        elevation_deg=0.0,
                        yaw_deg=0.0,
                        pitch_deg=0.0,
                        distance_m=dist_m,
                        conf=0.85,
                        cls_name="goal",
                        source_id=GOAL_SOURCE_ID,
                    )

            # Sync IMU yaw
            fake_imu.yaw_deg = yaw

            time.sleep(0.1)

    audio_thread = threading.Thread(target=audio_update_loop, daemon=True)
    audio_thread.start()

    # ---- Main game loop ----
    while running:
        dt = clock.tick(60) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_q]:
            running = False

        if not won:
            # Compute movement direction from yaw
            rad = math.radians(player_yaw)
            fwd_row = -math.cos(rad)   # north is -row direction
            fwd_col =  math.sin(rad)

            new_row, new_col = player_row, player_col

            if keys[pygame.K_w] or keys[pygame.K_UP]:
                new_row += fwd_row * MOVE_STEP
                new_col += fwd_col * MOVE_STEP
            if keys[pygame.K_s] or keys[pygame.K_DOWN]:
                new_row -= fwd_row * MOVE_STEP
                new_col -= fwd_col * MOVE_STEP
            if keys[pygame.K_a] or keys[pygame.K_LEFT]:
                player_yaw -= TURN_STEP
            if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
                player_yaw += TURN_STEP

            player_yaw %= 360

            # Collision: only move if new cell is not a wall
            if not cell_is_wall(int(new_row), int(new_col)):
                player_row, player_col = new_row, new_col

            # Check win
            if cell_is_goal(int(player_row), int(player_col)):
                won = True
                print("[MAZE] You reached the goal! Well done.")

        # ---- Draw ----
        screen.fill(CLR_OPEN)

        for r in range(ROWS):
            for c in range(COLS):
                x = c * CELL_SIZE
                y = r * CELL_SIZE
                v = MAZE[r][c]
                if v == 1:
                    pygame.draw.rect(screen, CLR_WALL, (x, y, CELL_SIZE, CELL_SIZE))
                elif v == 2:
                    pygame.draw.rect(screen, CLR_GOAL, (x, y, CELL_SIZE, CELL_SIZE))

        # Player circle
        px = int(player_col * CELL_SIZE)
        py = int(player_row * CELL_SIZE)
        pygame.draw.circle(screen, CLR_PLAYER, (px, py), CELL_SIZE // 3)

        # Heading arrow
        rad = math.radians(player_yaw)
        arrow_len = CELL_SIZE // 2
        ex = int(px + math.sin(rad)  * arrow_len)
        ey = int(py - math.cos(rad)  * arrow_len)
        pygame.draw.line(screen, CLR_ARROW, (px, py), (ex, ey), 3)

        # HUD
        hud = font.render(
            f"Pos: ({player_row:.1f}, {player_col:.1f})  Yaw: {player_yaw:.0f}°  {'🎉 GOAL!' if won else ''}",
            True, CLR_TEXT
        )
        screen.blit(hud, (8, 8))
        if won:
            big = pygame.font.SysFont("monospace", 36).render("YOU WIN!", True, (30, 160, 80))
            screen.blit(big, (COLS * CELL_SIZE // 2 - 80, ROWS * CELL_SIZE // 2 - 20))

        pygame.display.flip()

    # ---- Shutdown ----
    running = False
    audio_thread.join(timeout=1.0)
    processor.stop_playback()
    pygame.quit()
    print("[MAZE] Goodbye.")


if __name__ == "__main__":
    main()

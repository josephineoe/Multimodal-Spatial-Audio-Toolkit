# ============================================================================
# MAZE DEMO — Multimodal Spatial Audio Toolkit  (FIXED)
# Replaces IMU + vision.py with keyboard controls + 2D top-down view.
#
# Controls: W/S = move forward/back, A/D = turn left/right, Q = quit
# Install:  pip install pygame numpy soundfile sounddevice scipy
#
# FIXES vs original:
#   1. Movement speed now frame-rate independent (dt-scaled)
#   2. Turn speed reasonable (120 deg/s held, not 600 deg/s)
#   3. Collision uses player radius, not just centre cell
#   4. Audio thread reads player state via a shared dict (no closure rebinding risk)
#   5. Maze is wider / more navigable (3-cell-wide corridors)
#   6. Added mini-map with field-of-view cone so orientation is obvious
# ============================================================================

import math
import threading
import time

import numpy as np
import pygame

import importlib.util, sys

HRTF_PATH = r"C:\Users\JosephineOE\Documents\Rizzo Lab\Multimodal-Spatial-Audio-Toolkit\hrtf.py"
spec = importlib.util.spec_from_file_location("hrtf", HRTF_PATH)
hrtf_mod = importlib.util.module_from_spec(spec)
sys.modules["hrtf"] = hrtf_mod
spec.loader.exec_module(hrtf_mod)
SpatialAudioProcessor = hrtf_mod.SpatialAudioProcessor

# ---------------------------------------------------------------------------
# Maze  (1=wall, 0=open, 2=goal)
# Made wider so corridors are 2 cells across — much easier to navigate
# ---------------------------------------------------------------------------
MAZE = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1],
    [1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1],
    [1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1],
    [1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1],
    [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 2, 1],  # 2 = goal
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
]

CELL_SIZE    = 52
ROWS         = len(MAZE)
COLS         = len(MAZE[0])

# Player movement constants
MOVE_SPEED   = 3.0    # cells per second (held key)
TURN_SPEED   = 120.0  # degrees per second (held key)
PLAYER_RADIUS = 0.3   # collision radius in cells

# Audio radii
WALL_RADIUS  = 4.0
GOAL_RADIUS  = 8.0

GOAL_SOURCE_ID = 1
WALL_SOURCE_ID = 0


# ---------------------------------------------------------------------------
# Fake IMU
# ---------------------------------------------------------------------------
class FakeIMU:
    def __init__(self):
        self.yaw_deg   = 0.0
        self.pitch_deg = 0.0
    def get_euler(self):
        return 0.0, self.pitch_deg, self.yaw_deg
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
    return MAZE[int(row)][int(col)] == 1

def cell_is_goal(row, col):
    if row < 0 or row >= ROWS or col < 0 or col >= COLS:
        return False
    return MAZE[int(row)][int(col)] == 2

def find_goal():
    for r in range(ROWS):
        for c in range(COLS):
            if MAZE[r][c] == 2:
                return (r + 0.5, c + 0.5)
    return None

def find_nearest_wall(player_row, player_col, radius):
    best, best_dist = None, radius + 1
    ri = int(radius) + 1
    for dr in range(-ri, ri + 1):
        for dc in range(-ri, ri + 1):
            nr = int(player_row) + dr
            nc = int(player_col) + dc
            if cell_is_wall(nr, nc):
                dist = math.sqrt((nr + 0.5 - player_row)**2 + (nc + 0.5 - player_col)**2)
                if dist < best_dist:
                    best_dist = dist
                    best = (nr + 0.5 - player_row, nc + 0.5 - player_col, best_dist)
    return best

def can_move_to(row, col):
    """Check a circle of radius PLAYER_RADIUS around (row,col) for walls."""
    r = PLAYER_RADIUS
    corners = [
        (row - r, col - r), (row - r, col + r),
        (row + r, col - r), (row + r, col + r),
        (row,     col),
    ]
    return all(not cell_is_wall(pr, pc) for pr, pc in corners)

def world_angle_to_azimuth(dx_col, dy_row, player_yaw_deg):
    world_angle = math.degrees(math.atan2(dx_col, -dy_row))
    rel = (world_angle - player_yaw_deg + 180) % 360 - 180
    return rel


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
def draw_fov_cone(surface, px, py, yaw_deg, color=(240, 180, 60, 80), fov=60, length=80):
    """Draw a translucent FOV cone showing where the player is facing."""
    left_ang  = math.radians(yaw_deg - fov / 2)
    right_ang = math.radians(yaw_deg + fov / 2)
    mid_ang   = math.radians(yaw_deg)
    points = [
        (px, py),
        (px + math.sin(left_ang)  * length, py - math.cos(left_ang)  * length),
        (px + math.sin(mid_ang)   * length, py - math.cos(mid_ang)   * length),
        (px + math.sin(right_ang) * length, py - math.cos(right_ang) * length),
    ]
    # Draw on a temp surface for alpha
    tmp = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
    pygame.draw.polygon(tmp, color, [(int(x), int(y)) for x, y in points])
    surface.blit(tmp, (0, 0))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    pygame.init()
    screen = pygame.display.set_mode((COLS * CELL_SIZE, ROWS * CELL_SIZE + 30))
    pygame.display.set_caption("Spatial Audio Maze — WASD to move, Q to quit")
    clock = pygame.time.Clock()
    font  = pygame.font.SysFont("monospace", 15)

    # Player state
    state = {
        "row": 1.5,
        "col": 1.5,
        "yaw": 0.0,   # 0=North, clockwise positive
    }

    # ---- Audio setup ----
    VISION_CONFIG = {
        "gate_conf_thres":             0.0,
        "enable_distance_attenuation": True,
        "distance_ref_m":              1.4,
        "no_detection_fade_s":         0.5,
        "gain_min":                    0.0,
        "gain_max":                    1.0,
    }

    print("[MAZE] Initializing audio processor...")
    processor = SpatialAudioProcessor(
        audio_files=["rain.wav", "drums.wav"],
        sofa_file="MIT_KEMAR_normal_pinna.sofa",
        sample_rate=44100,
        imu_port=5005,
        vision_config=VISION_CONFIG,
    )
    fake_imu = FakeIMU()
    processor.imu = fake_imu
    processor.start_playback()
    print("[MAZE] Audio running. WASD / arrows to navigate.\n")

    goal_pos = find_goal()

    # ---- Colours ----
    CLR_WALL   = (55,  55,  75)
    CLR_OPEN   = (215, 215, 225)
    CLR_GOAL   = (70,  200, 110)
    CLR_PLAYER = (230, 90,  50)
    CLR_TEXT   = (20,  20,  45)
    CLR_BAR    = (30,  30,  55)

    running = True
    won     = False

    # ---- Audio thread reads from shared `state` dict (no rebinding issues) ----
    def audio_update_loop():
        while running:
            pr  = state["row"]
            pc  = state["col"]
            yaw = state["yaw"]

            # Nearest wall
            wall = find_nearest_wall(pr, pc, WALL_RADIUS)
            if wall:
                drow, dcol, dist_cells = wall
                az    = world_angle_to_azimuth(dcol, drow, yaw)
                dist_m = max(dist_cells * 0.5, 0.2)
                processor.update_vision_target(
                    azimuth_deg=az, elevation_deg=0.0,
                    yaw_deg=0.0, pitch_deg=0.0,
                    distance_m=dist_m, conf=0.9,
                    cls_name="wall", source_id=WALL_SOURCE_ID,
                )
            else:
                with processor._source_states_lock:
                    processor.source_states[WALL_SOURCE_ID].active = False

            # Goal beacon
            if goal_pos:
                grow, gcol = goal_pos
                drow = grow - pr
                dcol = gcol - pc
                dist_cells = math.sqrt(drow**2 + dcol**2)
                if dist_cells <= GOAL_RADIUS:
                    az    = world_angle_to_azimuth(dcol, drow, yaw)
                    dist_m = max(dist_cells * 0.5, 0.2)
                    processor.update_vision_target(
                        azimuth_deg=az, elevation_deg=0.0,
                        yaw_deg=0.0, pitch_deg=0.0,
                        distance_m=dist_m, conf=0.85,
                        cls_name="goal", source_id=GOAL_SOURCE_ID,
                    )

            fake_imu.yaw_deg = yaw
            time.sleep(0.05)   # 20 Hz audio updates

    audio_thread = threading.Thread(target=audio_update_loop, daemon=True)
    audio_thread.start()

    # ---- Main loop ----
    while running:
        dt = clock.tick(60) / 1000.0   # seconds since last frame

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_q]:
            running = False

        if not won:
            yaw = state["yaw"]
            rad = math.radians(yaw)
            fwd_row = -math.cos(rad)
            fwd_col =  math.sin(rad)

            # --- Turn (frame-rate independent) ---
            if keys[pygame.K_a] or keys[pygame.K_LEFT]:
                state["yaw"] = (yaw - TURN_SPEED * dt) % 360
            if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
                state["yaw"] = (yaw + TURN_SPEED * dt) % 360

            # --- Move (frame-rate independent, sliding collision) ---
            move = MOVE_SPEED * dt
            new_row = state["row"]
            new_col = state["col"]

            if keys[pygame.K_w] or keys[pygame.K_UP]:
                new_row += fwd_row * move
                new_col += fwd_col * move
            if keys[pygame.K_s] or keys[pygame.K_DOWN]:
                new_row -= fwd_row * move
                new_col -= fwd_col * move

            # Sliding collision: try full move, then axis-only fallbacks
            if can_move_to(new_row, new_col):
                state["row"], state["col"] = new_row, new_col
            elif can_move_to(new_row, state["col"]):   # slide along col axis
                state["row"] = new_row
            elif can_move_to(state["row"], new_col):   # slide along row axis
                state["col"] = new_col
            # else: fully blocked, don't move

            if cell_is_goal(state["row"], state["col"]):
                won = True
                print("[MAZE] 🎉 You reached the goal!")

        # ---- Draw ----
        # Status bar background
        screen.fill(CLR_BAR)

        # Grid
        for r in range(ROWS):
            for c in range(COLS):
                x = c * CELL_SIZE
                y = r * CELL_SIZE + 30
                v = MAZE[r][c]
                color = CLR_WALL if v == 1 else (CLR_GOAL if v == 2 else CLR_OPEN)
                pygame.draw.rect(screen, color, (x, y, CELL_SIZE, CELL_SIZE))
                # Light grid lines on open cells
                if v != 1:
                    pygame.draw.rect(screen, (190, 190, 200), (x, y, CELL_SIZE, CELL_SIZE), 1)

        # FOV cone
        px = int(state["col"] * CELL_SIZE)
        py = int(state["row"] * CELL_SIZE) + 30
        draw_fov_cone(screen, px, py, state["yaw"])

        # Player circle + direction arrow
        pygame.draw.circle(screen, CLR_PLAYER, (px, py), CELL_SIZE // 3)
        rad = math.radians(state["yaw"])
        arrow_len = CELL_SIZE // 2
        ex = int(px + math.sin(rad)  * arrow_len)
        ey = int(py - math.cos(rad)  * arrow_len)
        pygame.draw.line(screen, (255, 255, 255), (px, py), (ex, ey), 3)

        # HUD
        hud_text = (
            f"  Pos ({state['row']:.1f}, {state['col']:.1f})  "
            f"Yaw {state['yaw']:.0f}°  "
            f"WASD=move  Q=quit"
            + ("   🎉 YOU WIN! Press Q" if won else "")
        )
        hud = font.render(hud_text, True, (200, 200, 210))
        screen.blit(hud, (4, 7))

        pygame.display.flip()

    running = False
    audio_thread.join(timeout=1.0)
    processor.stop_playback()
    pygame.quit()
    print("[MAZE] Goodbye.")


if __name__ == "__main__":
    main()
# ============================================================================
# MAZE 3D — Multimodal Spatial Audio Toolkit Demo
# First-person raycaster + HRTF spatial audio (hrtf.py unchanged)
#
# Controls:
#   W / UP       — move forward
#   S / DOWN     — move back
#   A / LEFT     — turn left
#   D / RIGHT    — turn right
#   Q / ESC      — quit
#   R            — toggle recording
#   M            — toggle minimap
#
# Requirements: pip install pygame numpy soundfile sounddevice scipy
# Place next to: hrtf.py, rain.wav, drums.wav, MIT_KEMAR_normal_pinna.sofa
# ============================================================================

import math
import os
import sys
import threading
import time

import numpy as np
import pygame
import pygame.gfxdraw

# ── Try to import your HRTF module ──────────────────────────────────────────
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from hrtf import SpatialAudioProcessor
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("[WARN] hrtf.py not found — running visuals-only mode")

# ============================================================================
# MAZE  (0=open, 1=wall, 2=goal, 3=pillar accent wall)
# ============================================================================
MAZE = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1],
    [1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1],
    [1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1],
    [1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1],
    [1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1],
    [1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
    [1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1],
    [1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 2, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
]

ROWS = len(MAZE)
COLS = len(MAZE[0])

# ============================================================================
# DISPLAY & RENDERING CONFIG
# ============================================================================
W, H        = 1024, 600
HALF_H      = H // 2
NUM_RAYS    = W                 # one ray per screen column for sharpness
FOV         = math.pi / 2.8    # ~64°  field of view
HALF_FOV    = FOV / 2
RAY_STEP    = FOV / NUM_RAYS
MAX_DEPTH   = 22.0
WALL_SCALE  = 1.1              # tweak wall height feel

# Movement
MOVE_SPEED  = 0.055
ROT_SPEED   = 0.040
PLAYER_R    = 0.25             # collision radius

# Audio
WALL_DETECT_R  = 5.0           # max cell distance for wall audio
GOAL_DETECT_R  = 12.0
AUDIO_HZ       = 10            # updates per second
WALL_SRC       = 0             # rain.wav  → nearest wall
GOAL_SRC       = 1             # drums.wav → goal beacon

# Minimap
MM_CELL     = 10
MM_PAD      = 12
MM_ALPHA    = 180

# ============================================================================
# COLOUR PALETTE  (dark sci-fi / utility aesthetic)
# ============================================================================
C_SKY_TOP   = (10,  12,  25)
C_SKY_BOT   = (30,  35,  70)
C_FLOOR_TOP = (55,  48,  38)
C_FLOOR_BOT = (20,  18,  14)
C_WALL_BASE = (160, 155, 180)   # normal wall tint
C_WALL_ACC  = (210, 130,  60)   # accent wall tint (type 3)
C_GOAL_GLOW = (60,  240, 130)   # goal wall tint
C_PLAYER    = (255, 100,  60)
C_HUD_BG    = (0,   0,   0,  160)
C_WHITE     = (245, 245, 255)
C_DIM       = (130, 130, 150)
C_BEACON    = (60,  240, 130)
C_WARN      = (240, 180,  50)

# ============================================================================
# UTILITIES
# ============================================================================
def lerp_color(a, b, t):
    t = max(0.0, min(1.0, t))
    return (
        int(a[0] + (b[0] - a[0]) * t),
        int(a[1] + (b[1] - a[1]) * t),
        int(a[2] + (b[2] - a[2]) * t),
    )

def is_wall(x, y):
    col, row = int(x), int(y)
    if row < 0 or row >= ROWS or col < 0 or col >= COLS:
        return True
    return MAZE[row][col] in (1, 3)

def wall_type(x, y):
    col, row = int(x), int(y)
    if row < 0 or row >= ROWS or col < 0 or col >= COLS:
        return 1
    return MAZE[row][col]

def find_cell(v):
    for r in range(ROWS):
        for c in range(COLS):
            if MAZE[r][c] == v:
                return (r + 0.5, c + 0.5)
    return None

def angle_diff(a, b):
    """Signed difference a - b, wrapped to [-π, π]."""
    d = (a - b + math.pi) % (2 * math.pi) - math.pi
    return d

# ============================================================================
# FAKE IMU  (drop-in replacement for HeadTrackingReceiver)
# ============================================================================
class FakeIMU:
    def __init__(self):
        self.yaw_deg   = 0.0
        self.pitch_deg = 0.0
        self.t_send    = time.time()

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

# ============================================================================
# RAYCASTER  — returns per-column (distance, wall_type, side) arrays
# ============================================================================
def cast_all_rays(px, py, angle):
    """Cast NUM_RAYS rays, return parallel arrays."""
    dists   = np.zeros(NUM_RAYS, dtype=np.float32)
    wtypes  = np.zeros(NUM_RAYS, dtype=np.uint8)
    sides   = np.zeros(NUM_RAYS, dtype=np.uint8)   # 0=N/S 1=E/W

    for i in range(NUM_RAYS):
        ray_a = angle - HALF_FOV + i * RAY_STEP
        sin_a = math.sin(ray_a)
        cos_a = math.cos(ray_a)

        # DDA — march in small steps
        depth = 0.05
        hit = False
        while depth < MAX_DEPTH:
            tx = px + cos_a * depth
            ty = py + sin_a * depth
            col, row = int(tx), int(ty)
            if row < 0 or row >= ROWS or col < 0 or col >= COLS:
                break
            cv = MAZE[row][col]
            if cv in (1, 2, 3):
                # Which face did we hit?  Compare fractional parts
                fx, fy = tx - col, ty - row
                # Closer to vertical edge (E/W face) or horizontal (N/S face)?
                if abs(fx - 0.5) > abs(fy - 0.5):
                    sides[i] = 1   # E/W
                else:
                    sides[i] = 0   # N/S
                dists[i]  = depth
                wtypes[i] = cv
                hit = True
                break
            depth += 0.04

        if not hit:
            dists[i]  = MAX_DEPTH
            wtypes[i] = 0

    # Fish-eye correction
    for i in range(NUM_RAYS):
        ray_a = angle - HALF_FOV + i * RAY_STEP
        dists[i] *= math.cos(ray_a - angle)

    return dists, wtypes, sides

# ============================================================================
# RENDERER
# ============================================================================
def render_frame(screen, surface_3d, px, py, angle,
                 goal_pos, beacon_phase, won):

    # ── Sky gradient (drawn into surface_3d) ────────────────────────────────
    for y in range(HALF_H):
        t = y / HALF_H
        c = lerp_color(C_SKY_TOP, C_SKY_BOT, t)
        pygame.draw.line(surface_3d, c, (0, y), (W, y))

    # ── Floor gradient ───────────────────────────────────────────────────────
    for y in range(HALF_H, H):
        t = (y - HALF_H) / HALF_H
        c = lerp_color(C_FLOOR_TOP, C_FLOOR_BOT, t)
        pygame.draw.line(surface_3d, c, (0, y), (W, y))

    # ── Raycasting ──────────────────────────────────────────────────────────
    dists, wtypes, sides = cast_all_rays(px, py, angle)

    for x in range(NUM_RAYS):
        dist   = max(dists[x], 0.1)
        wt     = wtypes[x]
        side   = sides[x]

        wall_h = min(int(H * WALL_SCALE / dist), H)
        y0 = HALF_H - wall_h // 2
        y1 = y0 + wall_h

        # Choose base colour by wall type
        if wt == 2:      # goal
            base = C_GOAL_GLOW
        elif wt == 3:    # accent
            base = C_WALL_ACC
        else:
            base = C_WALL_BASE

        # Distance fog
        fog  = 1.0 / (1.0 + dist * dist * 0.06)
        dark = 0.6 if side == 1 else 1.0   # N/S face slightly darker

        r = int(base[0] * fog * dark)
        g = int(base[1] * fog * dark)
        b = int(base[2] * fog * dark)
        color = (max(0,min(255,r)), max(0,min(255,g)), max(0,min(255,b)))

        # Goal wall pulses
        if wt == 2:
            pulse = 0.75 + 0.25 * math.sin(beacon_phase * 3.0)
            color = (
                int(color[0] * pulse),
                min(255, int(color[1] * pulse * 1.1)),
                int(color[2] * pulse),
            )

        pygame.draw.line(surface_3d, color, (x, y0), (x, y1))

        # Thin bright top/bottom edge on walls
        edge_c = lerp_color(color, C_WHITE, 0.25)
        if 0 <= y0 < H:
            surface_3d.set_at((x, y0), edge_c)
        if 0 <= y1 < H:
            surface_3d.set_at((x, y1), edge_c)

    screen.blit(surface_3d, (0, 0))

# ============================================================================
# MINIMAP
# ============================================================================
def render_minimap(screen, px, py, angle, show):
    if not show:
        return

    mw = COLS * MM_CELL
    mh = ROWS * MM_CELL
    ox = MM_PAD
    oy = H - mh - MM_PAD

    # Semi-transparent backing
    surf = pygame.Surface((mw, mh), pygame.SRCALPHA)
    surf.fill((0, 0, 0, MM_ALPHA))
    screen.blit(surf, (ox, oy))

    for r in range(ROWS):
        for c in range(COLS):
            v = MAZE[r][c]
            if v == 1:
                color = (180, 175, 200, 255)
            elif v == 3:
                color = (210, 130, 60, 255)
            elif v == 2:
                color = (60, 240, 130, 255)
            else:
                continue
            pygame.draw.rect(screen, color[:3],
                             (ox + c * MM_CELL + 1, oy + r * MM_CELL + 1,
                              MM_CELL - 1, MM_CELL - 1))

    # Player dot
    mx = ox + int(px * MM_CELL)
    my = oy + int(py * MM_CELL)
    pygame.draw.circle(screen, C_PLAYER, (mx, my), 4)

    # Heading arrow
    al = 12
    ex = int(mx + math.cos(angle) * al)
    ey = int(my + math.sin(angle) * al)
    pygame.draw.line(screen, C_PLAYER, (mx, my), (ex, ey), 2)

# ============================================================================
# HUD
# ============================================================================
def render_hud(screen, fonts, px, py, angle, goal_pos, won, audio_ok):
    font_sm = fonts["sm"]
    font_md = fonts["md"]
    font_lg = fonts["lg"]

    if won:
        txt = font_lg.render("YOU FOUND IT", True, C_BEACON)
        r   = txt.get_rect(center=(W // 2, H // 2 - 40))
        screen.blit(txt, r)
        sub = font_md.render("Press Q to exit", True, C_DIM)
        screen.blit(sub, sub.get_rect(center=(W // 2, H // 2 + 10)))
        return

    # Crosshair
    cx, cy = W // 2, HALF_H
    pygame.draw.line(screen, (255, 255, 255, 160), (cx - 10, cy), (cx + 10, cy), 1)
    pygame.draw.line(screen, (255, 255, 255, 160), (cx, cy - 10), (cx, cy + 10), 1)

    # Goal bearing indicator at top-center
    if goal_pos:
        grow, gcol = goal_pos
        dx = gcol - px
        dy = grow - py
        dist = math.sqrt(dx * dx + dy * dy)
        world_a = math.atan2(dy, dx)
        rel_a   = angle_diff(world_a, angle)
        bearing = math.degrees(rel_a)

        # Clamp arrow direction to screen edges if off-FOV
        half_fov_deg = math.degrees(HALF_FOV)
        if abs(bearing) <= half_fov_deg:
            # Inside FOV: draw at correct screen x
            sx = int(W / 2 + (bearing / half_fov_deg) * W / 2)
            sx = max(20, min(W - 20, sx))
            pygame.draw.polygon(screen, C_BEACON,
                                [(sx, 28), (sx - 8, 14), (sx + 8, 14)])
        else:
            # Outside FOV: show arrow on left/right edge
            side_x = W - 24 if bearing > 0 else 24
            arrow_pts = (
                [(side_x, H//2), (side_x-10, H//2-12), (side_x-10, H//2+12)]
                if bearing > 0 else
                [(side_x, H//2), (side_x+10, H//2-12), (side_x+10, H//2+12)]
            )
            pygame.draw.polygon(screen, C_BEACON, arrow_pts)

        # Distance readout
        label = font_sm.render(f"GOAL  {dist:.1f}m", True, C_BEACON)
        screen.blit(label, (W // 2 - label.get_width() // 2, 38))

    # Bottom bar
    bar_h = 28
    bar_surf = pygame.Surface((W, bar_h), pygame.SRCALPHA)
    bar_surf.fill((0, 0, 0, 140))
    screen.blit(bar_surf, (0, H - bar_h))

    controls = "W/S move   A/D turn   M map   R rec   Q quit"
    ctrl_txt = font_sm.render(controls, True, C_DIM)
    screen.blit(ctrl_txt, (10, H - bar_h + 7))

    audio_str = "AUDIO ON" if audio_ok else "AUDIO OFF"
    a_color   = C_BEACON if audio_ok else C_WARN
    at = font_sm.render(audio_str, True, a_color)
    screen.blit(at, (W - at.get_width() - 10, H - bar_h + 7))

# ============================================================================
# AUDIO BRIDGE
# ============================================================================
def find_nearest_wall_audio(pr, pc, radius):
    best_dist = radius + 1
    best = None
    ri = int(radius) + 1
    for dr in range(-ri, ri + 1):
        for dc in range(-ri, ri + 1):
            nr, nc = int(pr + dr), int(pc + dc)
            if 0 <= nr < ROWS and 0 <= nc < COLS and MAZE[nr][nc] in (1, 3):
                dist = math.sqrt(dr * dr + dc * dc)
                if dist < best_dist:
                    best_dist = dist
                    best = (nr + 0.5 - pr, nc + 0.5 - pc, best_dist)
    return best

def world_to_azimuth(dx_col, dy_row, player_yaw_deg):
    """Object offset → azimuth relative to player heading (degrees)."""
    world_angle_deg = math.degrees(math.atan2(dx_col, -dy_row))
    rel = (world_angle_deg - player_yaw_deg + 180) % 360 - 180
    return rel

# ============================================================================
# MAIN
# ============================================================================
def main():
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Spatial Audio Maze 3D")
    pygame.mouse.set_visible(True)
    clock = pygame.time.Clock()

    # Pre-render surface for 3D (avoid per-frame alloc)
    surface_3d = pygame.Surface((W, H))

    # Fonts
    fonts = {
        "sm": pygame.font.SysFont("monospace", 13),
        "md": pygame.font.SysFont("monospace", 20),
        "lg": pygame.font.SysFont("monospace", 36),
    }

    # ── Player state ──────────────────────────────────────────────────────
    start = find_cell(0)  # first open cell centre
    if start is None:
        start = (1.5, 1.5)
    player_row, player_col = start
    player_angle = 0.0    # radians, 0 = East

    goal_pos  = find_cell(2)
    won       = False
    show_map  = True
    beacon_ph = 0.0

    # ── Audio setup ───────────────────────────────────────────────────────
    processor  = None
    fake_imu   = None
    audio_live = False

    if AUDIO_AVAILABLE:
        VISION_CONFIG = {
            "gate_conf_thres":             0.0,
            "enable_distance_attenuation": True,
            "distance_ref_m":              1.4,
            "no_detection_fade_s":         0.4,
            "gain_min":                    0.0,
            "gain_max":                    1.0,
        }
        try:
            print("[AUDIO] Initializing SpatialAudioProcessor…")
            processor = SpatialAudioProcessor(
                audio_files=["rain.wav", "drums.wav"],
                sofa_file="MIT_KEMAR_normal_pinna.sofa",
                sample_rate=44100,
                imu_port=5005,
                vision_config=VISION_CONFIG,
            )
            fake_imu          = FakeIMU()
            processor.imu     = fake_imu
            processor.start_playback()
            audio_live        = True
            print("[AUDIO] Running.")
        except Exception as e:
            print(f"[AUDIO] Could not start: {e}")
            audio_live = False

    # ── Audio update thread ───────────────────────────────────────────────
    running_flag = [True]

    def audio_loop():
        while running_flag[0]:
            if not audio_live:
                time.sleep(0.1)
                continue

            pr   = player_row
            pc   = player_col
            yaw  = math.degrees(player_angle)

            # Wall source
            wall = find_nearest_wall_audio(pr, pc, WALL_DETECT_R)
            if wall:
                drow, dcol, dist_c = wall
                az    = world_to_azimuth(dcol, drow, yaw)
                dist_m = max(dist_c * 0.6, 0.2)
                processor.update_vision_target(
                    azimuth_deg=az, elevation_deg=0.0,
                    yaw_deg=0.0, pitch_deg=0.0,
                    distance_m=dist_m, conf=0.9,
                    cls_name="wall", source_id=WALL_SRC,
                )
            else:
                with processor._source_states_lock:
                    processor.source_states[WALL_SRC].active = False

            # Goal source
            if goal_pos and not won:
                grow, gcol = goal_pos
                drow = grow - pr
                dcol = gcol - pc
                dist_c = math.sqrt(drow * drow + dcol * dcol)
                if dist_c <= GOAL_DETECT_R:
                    az     = world_to_azimuth(dcol, drow, yaw)
                    dist_m = max(dist_c * 0.6, 0.2)
                    processor.update_vision_target(
                        azimuth_deg=az, elevation_deg=0.0,
                        yaw_deg=0.0, pitch_deg=0.0,
                        distance_m=dist_m, conf=0.85,
                        cls_name="goal", source_id=GOAL_SRC,
                    )

            # Sync IMU yaw
            fake_imu.yaw_deg = yaw

            time.sleep(1.0 / AUDIO_HZ)

    if audio_live:
        t = threading.Thread(target=audio_loop, daemon=True)
        t.start()

    # ── Main loop ─────────────────────────────────────────────────────────
    while True:
        dt = clock.tick(60) / 1000.0
        beacon_ph += dt

        # Events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running_flag[0] = False
                if processor: processor.stop_playback()
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running_flag[0] = False
                    if processor: processor.stop_playback()
                    pygame.quit()
                    sys.exit()
                if event.key == pygame.K_m:
                    show_map = not show_map
                if event.key == pygame.K_r and processor:
                    processor.toggle_recording()

        # Movement
        keys = pygame.key.get_pressed()
        if not won:
            turn = 0.0
            if keys[pygame.K_LEFT]  or keys[pygame.K_a]: turn -= 1.0
            if keys[pygame.K_RIGHT] or keys[pygame.K_d]: turn += 1.0
            player_angle += turn * ROT_SPEED

            fwd = 0.0
            if keys[pygame.K_UP]   or keys[pygame.K_w]: fwd += 1.0
            if keys[pygame.K_DOWN] or keys[pygame.K_s]: fwd -= 1.0

            if fwd != 0.0:
                nx = player_col + math.cos(player_angle) * fwd * MOVE_SPEED
                ny = player_row + math.sin(player_angle) * fwd * MOVE_SPEED

                # Slide-based collision
                if not is_wall(nx, player_row): player_col = nx
                if not is_wall(player_col, ny): player_row = ny

            # Win condition
            if goal_pos:
                grow, gcol = goal_pos
                if abs(player_row - grow) < 0.7 and abs(player_col - gcol) < 0.7:
                    won = True
                    print("[MAZE] Goal reached!")

        # Render
        render_frame(screen, surface_3d,
                     player_col, player_row, player_angle,
                     goal_pos, beacon_ph, won)
        render_minimap(screen, player_col, player_row, player_angle, show_map)
        render_hud(screen, fonts,
                   player_col, player_row, player_angle,
                   goal_pos, won, audio_live)

        pygame.display.flip()


if __name__ == "__main__":
    main()

"""
Futuristic Elite Sci-Fi Drag & Drop
- Full theme: glass panels, neon corners, pulsing glow, HUD, trails, particles, ripple
- Stable: guards against out-of-frame ROIs and camera issues
"""

import cv2
import numpy as np
import mediapipe as mp
import math
import time
import random
from collections import deque

# ---------------- CONFIG ------------------
BOX_SIZE = 170
PINCH_THRESHOLD = 40
SMOOTHING = 0.16

NEON = (0, 255, 255)         # neon color
GRID_COLOR = (28, 28, 34)    # dark grid
GRID_ALPHA = 0.20

POSITIONS = [(160,130), (420,130), (680,130),
             (160,340), (420,340), (680,340)]

OBJECTS = [{"pos": list(p), "size": BOX_SIZE} for p in POSITIONS]

TRAIl_MAX = 20   # number of points in neon trail
PARTICLE_LIMIT = 120
RIPPLE_LIFETIME = 0.6  # seconds

# ---------- safety / visual params ----------
FPS_SMOOTH = 0.9
# --------------------------------------------

# Mediapipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1,
                       min_detection_confidence=0.6,
                       min_tracking_confidence=0.6)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: cannot open camera. Try changing index in cv2.VideoCapture().")
    exit(1)

# particle system structures
particles = []  # each: dict{x,y,vx,vy,life,max_life,color,size}
trail = deque(maxlen=TRAIl_MAX)
ripples = []    # each: dict{x,y,start_time,max_radius}

selected = None
last_pinch = False
last_time = time.time()
frame_time = 1/30.0

# ----------------- util functions -----------------
def distance(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def safe_roi(x, y, s, W, H, min_sz=3):
    x0 = max(0, int(round(x)))
    y0 = max(0, int(round(y)))
    x1 = min(W, int(round(x + s)))
    y1 = min(H, int(round(y + s)))
    if x1 - x0 < min_sz or y1 - y0 < min_sz:
        return None
    return (x0, y0, x1, y1)

def draw_grid(frame):
    overlay = frame.copy()
    H, W = frame.shape[:2]
    step = 40
    for x in range(0, W, step):
        cv2.line(overlay, (x,0), (x,H), GRID_COLOR, 1)
    for y in range(0, H, step):
        cv2.line(overlay, (0,y), (W,y), GRID_COLOR, 1)
    return cv2.addWeighted(overlay, GRID_ALPHA, frame, 1-GRID_ALPHA, 0)

def draw_glass_box(frame, x, y, s):
    H, W = frame.shape[:2]
    roi = safe_roi(x, y, s, W, H)
    if roi is None:
        return
    x0, y0, x1, y1 = roi
    region = frame[y0:y1, x0:x1]
    # odd kernel <= min dim
    min_dim = min(region.shape[:2])
    k = min(51, (min_dim//2)|1)
    k = max(3, k)
    try:
        blurred = cv2.GaussianBlur(region, (k, k), 0)
    except:
        blurred = region.copy()
    white = np.full_like(blurred, 245)
    glass = cv2.addWeighted(blurred, 0.72, white, 0.12, 0)
    # slight vignette (darken edges)
    frame[y0:y1, x0:x1] = cv2.addWeighted(frame[y0:y1, x0:x1], 0.15, glass, 0.85, 0)

def neon_corners(frame, x, y, s, t=3, glow_strength=0.12, pulse=1.0):
    # pulse controls line brightness and glow thickness
    line = int(28 * pulse)
    glow_thick = int(18 * (0.6 + 0.4*pulse))
    # glow layer
    glow = np.zeros_like(frame)
    pts = [
        ((x, y), (x+line, y)), ((x, y), (x, y+line)),
        ((x+s, y), (x+s-line, y)), ((x+s, y), (x+s, y+line)),
        ((x, y+s), (x+line, y+s)), ((x, y+s), (x, y+s-line)),
        ((x+s, y+s), (x+s-line, y+s)), ((x+s, y+s), (x+s, y+s-line)),
    ]
    for a,b in pts:
        cv2.line(glow, a, b, NEON, glow_thick)
    # merge smaller alpha
    frame[:] = cv2.addWeighted(frame, 1.0, glow, glow_strength, 0)
    # draw crisp corners
    for a,b in pts:
        cv2.line(frame, a, b, NEON, t)

def add_particles_at(x, y, intensity=8):
    if len(particles) > PARTICLE_LIMIT:
        return
    for _ in range(intensity):
        angle = random.uniform(0, 2*math.pi)
        speed = random.uniform(1.0, 5.2)
        vx = math.cos(angle) * speed
        vy = math.sin(angle) * speed
        size = random.randint(2,6)
        color = (int(NEON[0]*0.8 + random.randint(0,30)),
                 int(NEON[1]*0.9 + random.randint(0,30)),
                 int(NEON[2]*0.6 + random.randint(0,30)))
        particles.append({
            "x": x + random.randint(-6,6),
            "y": y + random.randint(-6,6),
            "vx": vx, "vy": vy,
            "life": 0.0, "max": random.uniform(0.5,1.1),
            "size": size, "color": color
        })

def update_particles(dt):
    i = 0
    while i < len(particles):
        p = particles[i]
        p["life"] += dt
        # simple physics + drag + gravity subtle
        p["vx"] *= 0.98
        p["vy"] *= 0.98
        p["vy"] += 0.02
        p["x"] += p["vx"]
        p["y"] += p["vy"]
        if p["life"] > p["max"]:
            particles.pop(i)
        else:
            i += 1

def render_particles(frame):
    for p in particles:
        alpha = max(0.0, 1.0 - p["life"]/p["max"])
        size = max(1, int(p["size"] * (0.6 + 0.4*alpha)))
        col = p["color"]
        # draw glow circle
        overlay = frame.copy()
        cv2.circle(overlay, (int(p["x"]), int(p["y"])), size*3, col, -1)
        frame[:] = cv2.addWeighted(overlay, 0.08*alpha, frame, 1-0.08*alpha, 0)
        cv2.circle(frame, (int(p["x"]), int(p["y"])), size, col, -1)

def add_ripple(x, y):
    ripples.append({"x": x, "y": y, "t": time.time(), "max_r": 160})

def update_render_ripples(frame):
    now = time.time()
    to_remove = []
    H, W = frame.shape[:2]
    for i, r in enumerate(ripples):
        age = now - r["t"]
        if age > RIPPLE_LIFETIME:
            to_remove.append(i)
            continue
        progress = age / RIPPLE_LIFETIME
        radius = int(r["max_r"] * progress)
        alpha = max(0.0, 1.0 - progress)
        thickness = max(1, int(6*(1-progress)))
        overlay = frame.copy()
        cv2.circle(overlay, (int(r["x"]), int(r["y"])), radius, NEON, thickness)
        frame[:] = cv2.addWeighted(overlay, 0.14*alpha, frame, 1-0.14*alpha, 0)
    # remove old
    for idx in sorted(to_remove, reverse=True):
        ripples.pop(idx)

def draw_neon_trail(frame, trail_pts):
    if len(trail_pts) < 2:
        return
    # draw fading polyline
    for i in range(1, len(trail_pts)):
        (x1,y1) = trail_pts[i-1]; (x2,y2) = trail_pts[i]
        alpha = (i / len(trail_pts))**1.2
        overlay = frame.copy()
        cv2.line(overlay, (int(x1),int(y1)), (int(x2),int(y2)), NEON, int(6*(0.6+alpha)))
        frame[:] = cv2.addWeighted(overlay, 0.08 + 0.6*alpha, frame, 1-(0.08 + 0.6*alpha), 0)

def hud_rings(frame, t):
    H, W = frame.shape[:2]
    cx, cy = W-120, 120
    overlay = frame.copy()
    # 3 rotating rings
    for i in range(3):
        radius = 30 + i*18 + int(6*math.sin(t*1.2 + i))
        thickness = 2
        ang = int((t*60 + i*40) % 360)
        # draw dashed-like arc (simulate rotation by drawing arcs at offsets)
        for a in range(0, 360, 24):
            start = (a + ang) % 360
            end = (start + 12) % 360
            cv2.ellipse(overlay, (cx,cy), (radius+ i*2, radius+ i*2), 0, start, end, NEON, thickness)
    frame[:] = cv2.addWeighted(overlay, 0.18, frame, 0.82, 0)

# ---------------- MAIN LOOP ----------------
print("Starting Elite Sci-Fi UI. Press ESC to exit.")
while True:
    t0 = time.time()
    ret, frame = cap.read()
    if not ret or frame is None:
        continue
    frame = cv2.flip(frame, 1)
    H, W = frame.shape[:2]

    # HUD grid
    frame = draw_grid(frame)

    # hand detect
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)

    pinch = False
    finger = None

    if res.multi_hand_landmarks:
        for hand in res.multi_hand_landmarks:
            lm = hand.landmark
            x_t, y_t = int(lm[4].x * W), int(lm[4].y * H)
            x_i, y_i = int(lm[8].x * W), int(lm[8].y * H)
            d = distance((x_t,y_t),(x_i,y_i))
            if d < PINCH_THRESHOLD:
                pinch = True
                finger = (x_i, y_i)
                cv2.circle(frame, finger, 10, NEON, -1)
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

    # pinch start => spawn ripple once
    if pinch and not last_pinch and finger is not None:
        add_ripple(finger[0], finger[1])
        add_particles_at(finger[0], finger[1], intensity=14)

    # dragging logic
    if pinch and finger:
        # update neon trail
        trail.appendleft(finger)
        if selected is None:
            for i, obj in enumerate(OBJECTS):
                ox, oy = obj["pos"]; s = obj["size"]
                if ox < finger[0] < ox+s and oy < finger[1] < oy+s:
                    selected = i
                    break
        else:
            ox, oy = OBJECTS[selected]["pos"]
            nx = int(ox + (finger[0] - ox) * SMOOTHING)
            ny = int(oy + (finger[1] - oy) * SMOOTHING)
            nx = max(0, min(W - BOX_SIZE, nx))
            ny = max(0, min(H - BOX_SIZE, ny))
            OBJECTS[selected]["pos"] = [nx, ny]
            # while dragging spawn particles continuously at lower intensity
            add_particles_at(finger[0], finger[1], intensity=3)
    else:
        # not pinching - slowly decay trail
        if len(trail) > 0:
            # pop a few to let it fade
            for _ in range(1):
                if len(trail)>0:
                    trail.pop()
        selected = None

    # update particles
    dt = max(1e-6, time.time() - t0)
    update_particles(dt)
    # draw boxes (shadow->glass->neon)
    for idx, obj in enumerate(OBJECTS):
        x, y = obj["pos"]; s = obj["size"]
        # shadow (soft rectangle)
        xsh, ysh = x+9, y+9
        roi_shadow = safe_roi(xsh, ysh, s, W, H)
        if roi_shadow:
            xs0, ys0, xs1, ys1 = roi_shadow
            shadow_region = frame[ys0:ys1, xs0:xs1].copy()
            dark = (shadow_region.astype(np.float32) * 0.6).astype(np.uint8)
            frame[ys0:ys1, xs0:xs1] = dark
        # glass panel
        draw_glass_box(frame, x, y, s)
        # compute pulsing based on time & index
        pulse = 0.85 + 0.35 * (0.5 + 0.5*math.sin(time.time()*2.0 + idx))
        neon_corners(frame, x, y, s, t=3, glow_strength=0.14, pulse=pulse)

    # render trail and particles and ripples and HUD rings
    draw_neon_trail(frame, list(trail))
    render_particles(frame)
    update_render_ripples(frame)
    hud_rings(frame, time.time()*0.8)

    cv2.putText(frame, "Elite Sci-Fi UI  ·  Pinch to move  ·  ESC to exit", (14,28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, NEON, 2, cv2.LINE_AA)

    cv2.imshow("Elite Sci-Fi Drag & Drop", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

    last_pinch = pinch

cap.release()
cv2.destroyAllWindows()

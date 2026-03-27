"""
sop_zone_main.py
────────────────────────────────────────────────────────────────────────────
SOP Assembly Verifier
  - MediaPipe Hands  →  zone detection + grip detection (dari test_hand_comparison)
  - DINOv2           →  scene similarity vs SOP reference image (dari main.py)

Alur per step:
  1. Tangan masuk ZONE PICK + grip terdeteksi  →  status "PICKED"
  2. Tangan masuk ZONE CHECK (setelah picked)   →  ambil frame, kirim ke DINOv2
  3. Similarity >= threshold                    →  PASS → lanjut step berikutnya
  4. Similarity <  threshold                    →  FAIL → operator ulangi

Urutan step DILOCK — step N hanya aktif setelah step N-1 PASS.

Hotkeys:
  Q / ESC  → quit
  S        → simpan snapshot
  R        → reset ke step 1
────────────────────────────────────────────────────────────────────────────
"""

import cv2
import mediapipe as mp
import numpy as np
import time
from icecream import ic

from DINOv2Encoder import DINOv2Encoder
from SOPReferenceBank import SOPReferenceBank
from SOPVerifier import SOPVerifier

# ── MediaPipe ─────────────────────────────────────────────────────────────────
mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# ── Landmark groups ───────────────────────────────────────────────────────────
FINGER_TIPS   = [4, 8, 12, 16, 20]
THUMB_FINGER  = [2, 3, 4]
INDEX_FINGER  = [5, 6, 7, 8]
MIDDLE_FINGER = [9, 10, 11, 12]

# ── Colours (BGR) ─────────────────────────────────────────────────────────────
CLR_WHITE  = (240, 240, 240)
CLR_GREEN  = (50,  220,  90)
CLR_RED    = (60,   60, 220)
CLR_YELLOW = (30,  220, 200)
CLR_ACCENT = (0,   210, 255)
CLR_ORANGE = (0,   140, 255)
CLR_GRAY   = (120, 120, 120)
CLR_PURPLE = (200,  80, 200)

# ── Tuning ────────────────────────────────────────────────────────────────────
GRIP_THRESHOLD  = 0.25   # normalized dist finger-tip → palm center
DWELL_CHECK_SEC = 0.6    # detik tangan harus di zone check sebelum trigger verify
SIM_THRESHOLD   = 0.80   # cosine similarity DINOv2 untuk PASS


# ══════════════════════════════════════════════════════════════════════════════
#  SOP STEP CONFIG  —  sesuaikan zone dan path gambar kamu di sini
# ══════════════════════════════════════════════════════════════════════════════

SOP_STEPS = [
    {
        "step_id"     : 0,
        "name"        : "STEP 1",
        "instruction" : "Ambil item dari area MERAH, pasang di area KUNING",
        "zone_pick"   : (246,  2, 453, 186),   # area ambil item
        "zone_check"  : (149, 163, 298, 368),   # area pasang / verifikasi
        "clr_pick"    : CLR_RED,
        "clr_check"   : CLR_YELLOW,
    },
    {
        "step_id"     : 1,
        "name"        : "STEP 2",
        "instruction" : "Ambil item dari area ORANGE, pasang di area HIJAU",
        "zone_pick"   : (365, 162, 494, 356),
        "zone_check"  : (50,  280, 230, 460),
        "clr_pick"    : CLR_ORANGE,
        "clr_check"   : CLR_GREEN,
    },
    {
        "step_id"     : 2,
        "name"        : "STEP 3",
        "instruction" : "Ambil item dari area UNGU, pasang di area CYAN",
        "zone_pick"   : (440, 10, 630, 180),
        "zone_check"  : (50,  10, 230, 180),
        "clr_pick"    : CLR_PURPLE,
        "clr_check"   : CLR_ACCENT,
    },
]


# ══════════════════════════════════════════════════════════════════════════════
#  HAND HELPERS  (dari test_hand_comparison, tidak diubah)
# ══════════════════════════════════════════════════════════════════════════════

def extract_hand_data(lms, label: str) -> dict:
    wrist      = (lms.landmark[0].x, lms.landmark[0].y)
    thumb_data = [[lms.landmark[i].x, lms.landmark[i].y] for i in THUMB_FINGER]
    index_data = [[lms.landmark[i].x, lms.landmark[i].y] for i in INDEX_FINGER]
    mid_data   = [[lms.landmark[i].x, lms.landmark[i].y] for i in MIDDLE_FINGER]
    return {"label": label, "wrist": wrist,
            "thumb_point": thumb_data, "index_point": index_data, "mid_point": mid_data}


def parse_to_array(hand: dict) -> np.ndarray:
    pts = []
    wx, wy = hand["wrist"]
    pts.append([wx, wy])
    for x, y in hand["thumb_point"]:  pts.append([x, y])
    for x, y in hand["index_point"]:  pts.append([x, y])
    for x, y in hand["mid_point"]:    pts.append([x, y])
    return np.array(pts, dtype=np.float32)


def embed(hand: dict) -> np.ndarray:
    pts    = parse_to_array(hand)
    center = (pts[5] + pts[9]) / 2
    pts   -= center
    scale  = np.linalg.norm(pts[5] - pts[9])
    if scale > 1e-6:
        pts /= scale
    base       = pts.flatten()
    thumb_tip  = pts[3];  index_tip = pts[7];  middle_tip = pts[11]
    d1 = np.linalg.norm(thumb_tip  - index_tip)
    d2 = np.linalg.norm(index_tip  - middle_tip)
    d3 = np.linalg.norm(thumb_tip  - middle_tip)
    return np.concatenate([base, np.array([d1, d2, d3])])


def is_grip(lms, threshold=GRIP_THRESHOLD) -> bool:
    pts    = np.array([[lm.x, lm.y] for lm in lms.landmark])
    center = (pts[0] + pts[5] + pts[17]) / 3
    return sum(np.linalg.norm(pts[t] - center) < threshold
               for t in [4, 8, 12, 16, 20]) >= 3


def wrist_pixel(lms, w, h):
    lm = lms.landmark[0]
    return int(lm.x * w), int(lm.y * h)


def fingertip_pixels(lms, w, h):
    return [(int(lms.landmark[t].x * w), int(lms.landmark[t].y * h))
            for t in FINGER_TIPS]


def point_in_zone(px, py, zone) -> bool:
    x1, y1, x2, y2 = zone
    return x1 <= px <= x2 and y1 <= py <= y2


def any_tip_in_zone(tips, zone) -> bool:
    return any(point_in_zone(cx, cy, zone) for cx, cy in tips)


# ══════════════════════════════════════════════════════════════════════════════
#  DRAWING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def draw_zone(frame, zone, color, label, active=False):
    x1, y1, x2, y2 = zone
    thickness = 3 if active else 1
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    # semi-transparent fill when active
    if active:
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
    cv2.putText(frame, label, (x1 + 4, y1 + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def draw_label(frame, text, color, x=14, y=100):
    tw = len(text) * 9 + 8
    cv2.rectangle(frame, (x - 2, y - 18), (x + tw, y + 6), (15, 15, 25), -1)
    cv2.putText(frame, text, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)


def draw_sim_bar(frame, sim, x=14, y=60, bw=200, bh=14):
    filled = int(bw * max(0.0, sim))
    color  = CLR_GREEN if sim >= SIM_THRESHOLD else CLR_ACCENT
    cv2.rectangle(frame, (x, y), (x + bw, y + bh), (50, 50, 50), -1)
    cv2.rectangle(frame, (x, y), (x + filled, y + bh), color, -1)
    cv2.putText(frame, f"sim: {sim:.3f}", (x + bw + 8, y + 11),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)


def draw_step_progress(frame, steps, current_step, w):
    """Top-right mini progress panel."""
    panel_x = w - 220
    for i, step in enumerate(steps):
        y = 20 + i * 28
        if i < current_step:
            color, symbol = CLR_GREEN,  "DONE"
        elif i == current_step:
            color, symbol = CLR_YELLOW, "NOW >"
        else:
            color, symbol = CLR_GRAY,   "LOCK"
        cv2.putText(frame, f"{step['name']} [{symbol}]", (panel_x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)


def draw_dwell_bar(frame, elapsed, total, x=14, y=170, bw=200, bh=10):
    """Progress bar untuk dwell timer."""
    ratio  = min(elapsed / total, 1.0)
    filled = int(bw * ratio)
    cv2.rectangle(frame, (x, y), (x + bw, y + bh), (50, 50, 50), -1)
    cv2.rectangle(frame, (x, y), (x + filled, y + bh), CLR_ACCENT, -1)
    cv2.putText(frame, "dwell...", (x + bw + 8, y + 9),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, CLR_ACCENT, 1, cv2.LINE_AA)


def flash_result(frame, passed: bool, message: str):
    color   = CLR_GREEN if passed else CLR_RED
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 70), color, -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
    cv2.putText(frame, message[:55], (10, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, CLR_WHITE, 2, cv2.LINE_AA)


# ══════════════════════════════════════════════════════════════════════════════
#  PER-STEP RUNTIME STATE  (simple dict, no state machine class)
# ══════════════════════════════════════════════════════════════════════════════

def make_step_runtime():
    return {
        "picked"       : False,   # True setelah grip di zone pick terkonfirmasi
        "check_entered": False,   # True saat tangan pertama kali masuk zone check
        "dwell_start"  : None,    # timestamp saat dwell di zone check dimulai
        "last_sim"     : 0.0,
        "passed"       : False,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # ── Init DINOv2 + reference bank ─────────────────────────────────────────
    print("[INFO] Loading DINOv2 encoder...")
    encoder = DINOv2Encoder(model_name="dinov2_vitb14")

    bank = SOPReferenceBank(encoder)
    # Ganti ke bank.load() kalau sudah pernah save
    bank.register_from_folder("image_test/SOP")
    bank.save("image_test/SOP")

    verifier = SOPVerifier(encoder=encoder, bank=bank,
                           pass_threshold=SIM_THRESHOLD)

    # ── Init MediaPipe ────────────────────────────────────────────────────────
    hands_model = mp_hands.Hands(
        model_complexity=1,
        max_num_hands=2,
        min_detection_confidence=0.55,
        min_tracking_confidence=0.50,
    )
    conn_style = mp_drawing.DrawingSpec(color=CLR_ACCENT, thickness=2)
    lm_style   = mp_drawing.DrawingSpec(color=CLR_GREEN,  thickness=4, circle_radius=4)

    # ── Camera ────────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # ── Runtime per step ──────────────────────────────────────────────────────
    runtimes     = [make_step_runtime() for _ in SOP_STEPS]
    current_step = 0          # index step yang aktif sekarang
    all_done     = False

    fps      = 0.0
    prev_t   = time.time()
    last_sim = 0.0

    print("\nHotkeys: Q/ESC = quit   S = snapshot   R = reset")
    print(f"Starting at {SOP_STEPS[0]['name']}: {SOP_STEPS[0]['instruction']}\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("[ERR] Frame grab failed")
            break

        frame        = cv2.flip(frame, 1)
        display      = frame.copy()
        h, w         = display.shape[:2]
        rgb          = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_result    = hands_model.process(rgb)

        step_cfg     = SOP_STEPS[current_step]
        rt           = runtimes[current_step]
        zone_pick    = step_cfg["zone_pick"]
        zone_check   = step_cfg["zone_check"]

        # ── Draw all zones ────────────────────────────────────────────────────
        for i, s in enumerate(SOP_STEPS):
            alpha = 1.0 if i == current_step else 0.4
            # dim inactive zones by drawing with gray
            zc_pick  = s["clr_pick"]  if i == current_step else CLR_GRAY
            zc_check = s["clr_check"] if i == current_step else CLR_GRAY
            label_pick  = f"PICK S{i+1}"  if i >= current_step else f"S{i+1} DONE"
            label_check = f"CHECK S{i+1}" if i >= current_step else f"S{i+1} DONE"
            draw_zone(display, s["zone_pick"],  zc_pick,  label_pick,
                      active=(i == current_step and not rt["picked"]))
            draw_zone(display, s["zone_check"], zc_check, label_check,
                      active=(i == current_step and rt["picked"]))

        # ── Progress panel ────────────────────────────────────────────────────
        draw_step_progress(display, SOP_STEPS, current_step, w)

        # ── Per-frame hand variables ──────────────────────────────────────────
        hand_in_pick  = False
        hand_in_check = False
        grip_detected = False
        live_embedding= None

        if mp_result.multi_hand_landmarks and mp_result.multi_handedness:
            for lms, handedness in zip(mp_result.multi_hand_landmarks,
                                       mp_result.multi_handedness):
                label = handedness.classification[0].label

                # Draw skeleton
                mp_drawing.draw_landmarks(display, lms, mp_hands.HAND_CONNECTIONS,
                                          lm_style, conn_style)

                # Fingertip pixels
                tips = fingertip_pixels(lms, w, h)
                for cx, cy in tips:
                    cv2.circle(display, (cx, cy), 9, CLR_YELLOW, -1)
                    cv2.circle(display, (cx, cy), 9, CLR_WHITE,  2)

                # Wrist pixel (for zone check — more stable than fingertip)
                wx_px, wy_px = wrist_pixel(lms, w, h)

                # Zone membership
                if any_tip_in_zone(tips, zone_pick):
                    hand_in_pick = True
                # Use wrist OR any tip for check zone
                if (point_in_zone(wx_px, wy_px, zone_check) or
                        any_tip_in_zone(tips, zone_check)):
                    hand_in_check = True

                # Grip
                if is_grip(lms):
                    grip_detected = True

                # Embedding (first hand only)
                if live_embedding is None:
                    hand_data     = extract_hand_data(lms, label)
                    live_embedding = embed(hand_data)

        # ── Grip status label ─────────────────────────────────────────────────
        if mp_result.multi_hand_landmarks:
            grip_text  = "GRIP" if grip_detected else "OPEN"
            grip_color = CLR_GREEN if grip_detected else CLR_YELLOW
            draw_label(display, grip_text, grip_color, x=14, y=140)

        # ══════════════════════════════════════════════════════════════════════
        #  PICK LOGIC
        #  Tangan di zone pick + grip → set picked = True
        # ══════════════════════════════════════════════════════════════════════
        if not all_done and not rt["picked"]:
            if hand_in_pick and grip_detected:
                rt["picked"] = True
                print(f"[PICK] {step_cfg['name']} — item picked")

        # ══════════════════════════════════════════════════════════════════════
        #  CHECK LOGIC
        #  Setelah picked: tangan masuk zone check → dwell → trigger DINOv2
        # ══════════════════════════════════════════════════════════════════════
        if not all_done and rt["picked"] and not rt["passed"]:

            if hand_in_check:
                # Start dwell timer
                if rt["dwell_start"] is None:
                    rt["dwell_start"] = time.time()
                    rt["check_entered"] = True
                    print(f"[CHECK] {step_cfg['name']} — hand in check zone, dwelling...")

                elapsed = time.time() - rt["dwell_start"]
                draw_dwell_bar(display, elapsed, DWELL_CHECK_SEC)

                # Dwell complete → trigger DINOv2 verify
                if elapsed >= DWELL_CHECK_SEC:
                    print(f"[VERIFY] Running DINOv2 for {step_cfg['name']}...")
                    verifier.jump_to_step(current_step)
                    result = verifier.verify(frame)
                    last_sim = result["similarity"]

                    flash_result(display, result["passed"], result["message"])
                    cv2.imshow("SOP Assembly", display)
                    cv2.waitKey(1200)

                    print(f"  → {result['message']}  sim={result['similarity']}")

                    if result["passed"]:
                        rt["passed"] = True
                        rt["last_sim"] = last_sim

                        # Advance ke step berikutnya
                        if current_step < len(SOP_STEPS) - 1:
                            current_step += 1
                            verifier.jump_to_step(current_step)
                            print(f"\n[NEXT] Proceeding to {SOP_STEPS[current_step]['name']}")
                            print(f"       {SOP_STEPS[current_step]['instruction']}")
                        else:
                            all_done = True
                            print("\n[DONE] All SOP steps completed!")
                    else:
                        # Reset dwell agar bisa coba lagi
                        rt["dwell_start"] = None
                        rt["picked"]      = False   # harus pick ulang
                        print(f"  → Retry: ambil item dan pasang kembali")

            else:
                # Tangan keluar zone check → reset dwell
                if rt["dwell_start"] is not None:
                    rt["dwell_start"] = None

        # ══════════════════════════════════════════════════════════════════════
        #  ALL DONE overlay
        # ══════════════════════════════════════════════════════════════════════
        if all_done:
            overlay = display.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 180, 60), -1)
            cv2.addWeighted(overlay, 0.25, display, 0.75, 0, display)
            cv2.putText(display, "ALL STEPS COMPLETE!", (60, h // 2 - 10),
                        cv2.FONT_HERSHEY_DUPLEX, 1.1, CLR_WHITE, 2, cv2.LINE_AA)
            cv2.putText(display, "Press R to reset", (160, h // 2 + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, CLR_YELLOW, 1, cv2.LINE_AA)

        # ── HUD ───────────────────────────────────────────────────────────────
        # Instruction bar
        inst_text = "ALL DONE — press R to reset" if all_done else step_cfg["instruction"]
        cv2.rectangle(display, (0, h - 30), (w, h), (15, 15, 25), -1)
        cv2.putText(display, inst_text, (8, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, CLR_WHITE, 1, cv2.LINE_AA)

        # Sim bar (last known)
        draw_sim_bar(display, last_sim)

        # Status: picked or not
        if not all_done:
            if rt["picked"]:
                draw_label(display, f"[{step_cfg['name']}] PICKED — go to CHECK zone",
                           CLR_GREEN, x=14, y=100)
            else:
                draw_label(display, f"[{step_cfg['name']}] Go to PICK zone + grip",
                           CLR_ACCENT, x=14, y=100)

        # FPS
        now   = time.time()
        fps   = 0.9 * fps + 0.1 / max(now - prev_t, 1e-6)
        prev_t = now
        cv2.putText(display, f"FPS {fps:.1f}", (14, 32),
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, CLR_GREEN, 1, cv2.LINE_AA)

        cv2.imshow("SOP Assembly", display)

        # ── Hotkeys ───────────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):
            break

        elif key == ord("s"):
            fname = f"snapshot_step{current_step + 1}_{int(time.time())}.png"
            cv2.imwrite(fname, frame)
            print(f"[SNAP] Saved {fname}")

        elif key == ord("r"):
            runtimes     = [make_step_runtime() for _ in SOP_STEPS]
            current_step = 0
            all_done     = False
            last_sim     = 0.0
            verifier.reset()
            print("\n[RESET] Back to Step 1")

    cap.release()
    cv2.destroyAllWindows()
    hands_model.close()


if __name__ == "__main__":
    main()
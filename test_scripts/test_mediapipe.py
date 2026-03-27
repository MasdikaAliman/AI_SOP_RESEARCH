"""
Real-Time Hand Pose Tracking using MediaPipe
============================================
Requirements:
    pip install mediapipe opencv-python numpy

Run:
    python hand_pose_tracking.py

Controls:
    Q / ESC  → Quit
    S        → Save screenshot
    M        → Toggle landmark IDs display
    F        → Toggle FPS display
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import os

# ─── MediaPipe Setup ──────────────────────────────────────────────────────────
mp_hands     = mp.solutions.hands
mp_drawing   = mp.solutions.drawing_utils
mp_draw_style = mp.solutions.drawing_styles

# ─── Finger / Joint Constants ─────────────────────────────────────────────────
FINGER_TIPS   = [4, 8, 12, 16, 20]           # thumb, index, middle, ring, pinky
FINGER_PIPS   = [3, 6, 10, 14, 18]
FINGER_NAMES  = ["Thumb", "Index", "Middle", "Ring", "Pinky"]

# ─── Colours (BGR) ────────────────────────────────────────────────────────────
CLR_BG        = (15,  15,  25)
CLR_ACCENT    = (0,  210, 255)   # cyan
CLR_GREEN     = (50,  220,  90)
CLR_RED       = (60,   60, 220)
CLR_WHITE     = (240, 240, 240)
CLR_YELLOW    = (30,  220, 200)
CLR_PANEL     = (25,  25,  40)


# ─── Helper: count raised fingers ────────────────────────────────────────────
def count_fingers(landmarks, handedness: str) -> list[bool]:
    """Return list of booleans [thumb, index, middle, ring, pinky] raised."""
    raised = []

    # Thumb: compare x-axis (mirrored for left/right hand)
    tip  = landmarks[FINGER_TIPS[0]]
    pip_ = landmarks[FINGER_PIPS[0]]
    if handedness == "Right":
        raised.append(tip.x < pip_.x)
    else:
        raised.append(tip.x > pip_.x)

    # Other four fingers: compare y-axis (tip above pip = raised)
    for i in range(1, 5):
        raised.append(landmarks[FINGER_TIPS[i]].y < landmarks[FINGER_PIPS[i]].y)

    return raised


# ─── Helper: gesture label ───────────────────────────────────────────────────
def get_gesture(raised: list[bool]) -> str:
    patterns = {
        (False, False, False, False, False): "✊ Fist",
        (True,  True,  True,  True,  True ): "🖐 Open Hand",
        (False, True,  False, False, False): "☝ Pointing",
        (False, True,  True,  False, False): "✌ Peace",
        (True,  False, False, False, False): "👍 Thumbs Up",
        (False, False, False, False, True ): "🤙 Pinky",
        (True,  True,  False, False, False): "🤞 Crossed",
        (False, True,  True,  True,  True ): "🤘 Rock",
        (True,  True,  True,  True,  False): "🤟 Love",
        (True,  False, False, False, True ): "🤙 Call Me",
        (False, True,  False, False, True ): "🤙 Spider-Man",
    }
    return patterns.get(tuple(raised), f"🖖 Custom ({sum(raised)} fingers)")


# ─── Helper: draw rounded rectangle ──────────────────────────────────────────
def draw_rounded_rect(img, pt1, pt2, color, radius=12, thickness=-1, alpha=0.6):
    overlay = img.copy()
    x1, y1 = pt1
    x2, y2 = pt2
    cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
    cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
    for cx, cy in [(x1+radius, y1+radius), (x2-radius, y1+radius),
                   (x1+radius, y2-radius), (x2-radius, y2-radius)]:
        cv2.circle(overlay, (cx, cy), radius, color, thickness)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


# ─── Helper: draw info panel ─────────────────────────────────────────────────
def draw_info_panel(frame, hand_data: list[dict], fps: float, show_fps: bool):
    h, w = frame.shape[:2]
    panel_w = 280
    px = w - panel_w - 10
    py = 10

    for idx, data in enumerate(hand_data[:2]):          # max 2 hands
        label     = data["label"]
        raised    = data["raised"]
        gesture   = data["gesture"]
        finger_ct = sum(raised)

        box_h = 185
        y_off = idx * (box_h + 10)

        draw_rounded_rect(frame,
                          (px, py + y_off),
                          (px + panel_w, py + y_off + box_h),
                          CLR_PANEL, radius=14, alpha=0.75)

        # Hand label
        cv2.putText(frame, f"{label} Hand", (px + 12, py + y_off + 28),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65, CLR_ACCENT, 1, cv2.LINE_AA)

        # Gesture
        cv2.putText(frame, gesture, (px + 12, py + y_off + 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, CLR_YELLOW, 1, cv2.LINE_AA)

        # Finger count
        cv2.putText(frame, f"Fingers: {finger_ct}",
                    (px + 12, py + y_off + 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, CLR_WHITE, 1, cv2.LINE_AA)

        # Per-finger indicators
        for fi, (name, up) in enumerate(zip(FINGER_NAMES, raised)):
            dot_clr = CLR_GREEN if up else CLR_RED
            bx = px + 12 + fi * 52
            by = py + y_off + 110
            cv2.circle(frame, (bx + 10, by), 9, dot_clr, -1)
            cv2.putText(frame, name[:3], (bx, by + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, CLR_WHITE, 1, cv2.LINE_AA)

        # Wrist coords
        wrist = data["wrist"]
        cv2.putText(frame, f"Wrist  x:{wrist[0]:.2f}  y:{wrist[1]:.2f}",
                    (px + 12, py + y_off + 158),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 200), 1, cv2.LINE_AA)

    # FPS badge
    if show_fps:
        draw_rounded_rect(frame, (8, 8), (115, 42), (10, 10, 10), alpha=0.7)
        cv2.putText(frame, f"FPS: {fps:.1f}", (14, 32),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65, CLR_GREEN, 1, cv2.LINE_AA)

    # Controls hint (bottom)
    hint = "Q/ESC: Quit  |  S: Screenshot  |  M: IDs  |  F: FPS"
    cv2.putText(frame, hint, (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 100, 130), 1, cv2.LINE_AA)


# ─── Helper: draw landmark IDs ───────────────────────────────────────────────
def draw_landmark_ids(frame, hand_landmarks, w, h):
    for i, lm in enumerate(hand_landmarks.landmark):
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.putText(frame, str(i), (cx - 6, cy - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, (255, 255, 255), 1, cv2.LINE_AA)


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)
    cap.set(cv2.CAP_PROP_FPS,           30)

    show_ids = False
    show_fps = True
    prev_t   = time.time()
    fps      = 0.0
    ss_count = 0

    custom_conn_style = mp_drawing.DrawingSpec(color=CLR_ACCENT,  thickness=2)
    custom_lm_style   = mp_drawing.DrawingSpec(color=CLR_GREEN,   thickness=4,
                                               circle_radius=4)

    with mp_hands.Hands(
        model_complexity=1,
        max_num_hands=2,
        min_detection_confidence=0.65,
        min_tracking_confidence=0.55,
    ) as hands:

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Cannot read from camera.")
                break

            frame = cv2.flip(frame, 1)          # mirror
            h, w  = frame.shape[:2]

            # Dark overlay tint
            overlay = np.full_like(frame, CLR_BG)
            frame   = cv2.addWeighted(frame, 0.85, overlay, 0.15, 0)

            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            hand_data = []

            if result.multi_hand_landmarks and result.multi_handedness:
                for lms, handedness in zip(result.multi_hand_landmarks,
                                           result.multi_handedness):
                    label  = handedness.classification[0].label   # "Left" / "Right"
                    raised = count_fingers(lms.landmark, label)
                    gesture = get_gesture(raised)
                    wrist  = (lms.landmark[0].x, lms.landmark[0].y)

                    hand_data.append({
                        "label":   label,
                        "raised":  raised,
                        "gesture": gesture,
                        "wrist":   wrist,
                    })

                    # Draw skeleton
                    mp_drawing.draw_landmarks(
                        frame, lms, mp_hands.HAND_CONNECTIONS,
                        custom_lm_style, custom_conn_style,
                    )

                    # Highlight fingertips
                    for tip_id in FINGER_TIPS:
                        lm = lms.landmark[tip_id]
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        cv2.circle(frame, (cx, cy), 10, CLR_YELLOW, -1)
                        cv2.circle(frame, (cx, cy), 10, CLR_WHITE,   2)

                    if show_ids:
                        draw_landmark_ids(frame, lms, w, h)

            # FPS
            now   = time.time()
            fps   = 0.9 * fps + 0.1 * (1.0 / max(now - prev_t, 1e-6))
            prev_t = now

            draw_info_panel(frame, hand_data, fps, show_fps)

            cv2.imshow("Hand Pose Tracking  •  MediaPipe", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):           # Q or ESC
                break
            elif key == ord('s'):               # Screenshot
                ss_count += 1
                fname = f"screenshot_{ss_count:03d}.png"
                cv2.imwrite(fname, frame)
                print(f"[SAVED] {fname}")
            elif key == ord('m'):
                show_ids = not show_ids
            elif key == ord('f'):
                show_fps = not show_fps

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
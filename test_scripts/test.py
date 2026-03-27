"""
assembly_monitor.py
===================
Real-time assembly step validator using MediaPipe Hands + OpenCV.

3-step demo scaffold:
  Step 0 — Pick up Component A  (Item Zone A → Assembly Zone → pose match)
  Step 1 — Pick up Component B  (Item Zone B → Assembly Zone → pose match)
  Step 2 — Final assembly pose  (Assembly Zone only → pose match)

Hotkeys (during runtime):
  R  — capture reference pose for the CURRENT step
  N  — force-advance to next step (debug)
  Q  — quit

Zone setup:
  • Pre-configure bboxes in ZONE_CONFIG below.
  • If a zone bbox is None, the program pauses on first launch and asks
    you to drag the zone with the mouse.

Dependencies:
    pip install opencv-python mediapipe numpy scipy
"""

import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import procrustes as scipy_procrustes
from enum import Enum, auto
from typing import Optional
import time


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Bounding boxes: (x1, y1, x2, y2) in pixel coords, or None to draw with mouse.
ZONE_CONFIG = {
    "item_a":   None,   # e.g. (30,  280, 180, 460)
    "item_b":   None,   # e.g. (460, 280, 620, 460)
    "assembly": None,   # e.g. (190, 120, 450, 380)
}

# 3-step demo
STEPS = [
    {
        "name":          "Pick up Component A",
        "item_zone":     "item_a",    # hand must visit this zone first
        "assembly_zone": "assembly",  # then enter here before pose check
        "ref_embedding": None,        # filled at runtime via R key
    },
    {
        "name":          "Pick up Component B",
        "item_zone":     "item_b",
        "assembly_zone": "assembly",
        "ref_embedding": None,
    },
    {
        "name":          "Final assembly pose",
        "item_zone":     None,        # no pickup zone for last step
        "assembly_zone": "assembly",
        "ref_embedding": None,
    },
]

# ---------- Tuning knobs ----------
SIMILARITY_THRESHOLD = 0.92   # cosine sim gate  [0..1], higher = stricter
HOLD_FRAMES          = 15     # frames pose must stay matched to confirm step
CAMERA_INDEX         = 0      # webcam device index


# ---------------------------------------------------------------------------
# State machine states
# ---------------------------------------------------------------------------

class State(Enum):
    IDLE             = auto()
    IN_ITEM_ZONE     = auto()
    IN_ASSEMBLY_ZONE = auto()
    POSE_MATCH       = auto()
    STEP_COMPLETE    = auto()


# ---------------------------------------------------------------------------
# PoseEstimator
# ---------------------------------------------------------------------------

class PoseEstimator:
    """
    Wraps MediaPipe Hands.
    Extracts 21 landmarks and normalises them to a scale-invariant vector.

    Normalisation math
    ------------------
    Given raw landmarks L = [(x0,y0), ..., (x20,y20)] in image coordinates:

    1. Translate:  L' = L - L[0]          (wrist moves to origin)
    2. Scale:      L'' = L' / max_dist     where max_dist = max |L'[i]|
    3. Flatten:    v = L''.flatten()       → 42-D vector

    This ensures the vector encodes *hand shape only*, independent of:
    - Distance from camera  (scale invariant)
    - Position in frame     (translation invariant)

    Cosine similarity on these vectors then measures how close two hand
    *shapes* are in 42-D angular space.
    """

    def __init__(self, max_hands: int = 1, min_detect_conf: float = 0.7):
        self.mp_hands = mp.solutions.hands
        self.mp_draw  = mp.solutions.drawing_utils
        self.hands    = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=min_detect_conf,
            min_tracking_confidence=0.6,
        )
        self.results = None

    def process(self, frame_rgb: np.ndarray):
        """Run MediaPipe on an RGB frame. Stores results internally."""
        self.results = self.hands.process(frame_rgb)

    def get_landmarks_px(self, frame_shape) -> Optional[np.ndarray]:
        """
        Returns (21, 2) float32 pixel array for the first detected hand,
        or None if no hand found.
        """
        if not self.results or not self.results.multi_hand_landmarks:
            return None
        h, w = frame_shape[:2]
        lm = self.results.multi_hand_landmarks[0].landmark
        return np.array([[p.x * w, p.y * h] for p in lm], dtype=np.float32)

    def normalize(self, landmarks_px: np.ndarray) -> np.ndarray:
        """
        Normalise (21, 2) landmark array → scale-invariant 42-D vector.
        """
        pts = landmarks_px.copy()
        pts -= pts[0]                                    # 1. translate to wrist
        max_dist = np.max(np.linalg.norm(pts, axis=1))
        if max_dist > 1e-6:
            pts /= max_dist                              # 2. unit scale
        return pts.flatten()                             # 3. flatten

    def draw_landmarks(self, frame_bgr: np.ndarray):
        """Draw MediaPipe landmark overlay on a BGR frame in-place."""
        if self.results and self.results.multi_hand_landmarks:
            for hand_lm in self.results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame_bgr,
                    hand_lm,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_draw.DrawingSpec(color=(0, 200, 255), thickness=2, circle_radius=3),
                    self.mp_draw.DrawingSpec(color=(0, 120, 200), thickness=2),
                )

    def get_hand_center_px(self, frame_shape) -> Optional[tuple]:
        """Returns (cx, cy) centroid of the detected hand, or None."""
        pts = self.get_landmarks_px(frame_shape)
        if pts is None:
            return None
        return (int(np.mean(pts[:, 0])), int(np.mean(pts[:, 1])))

    def close(self):
        self.hands.close()


# ---------------------------------------------------------------------------
# ZoneManager
# ---------------------------------------------------------------------------

class Zone:
    __slots__ = ("name", "bbox", "color")
    def __init__(self, name, bbox, color):
        self.name  = name
        self.bbox  = bbox    # (x1, y1, x2, y2)
        self.color = color   # BGR


class ZoneManager:
    """
    Manages named rectangular zones.
    Supports runtime mouse-draw for zones not pre-configured.
    """

    COLORS = {
        "item_a":   (30,  200, 255),   # yellow
        "item_b":   (255, 140,  30),   # blue-orange
        "assembly": (80,  255, 120),   # green
    }

    def __init__(self):
        self.zones: dict[str, Zone] = {}
        self._drawing    = False
        self._draw_start = None
        self._cur_rect   = None

    def add_zone(self, name: str, bbox: tuple):
        color = self.COLORS.get(name, (200, 200, 200))
        self.zones[name] = Zone(name, bbox, color)

    def point_in_zone(self, name: str, pt: tuple) -> bool:
        if name not in self.zones:
            return False
        x1, y1, x2, y2 = self.zones[name].bbox
        px, py = pt
        return x1 <= px <= x2 and y1 <= py <= y2

    def draw_zones(self, frame: np.ndarray):
        """Semi-transparent zone rectangles drawn on frame in-place."""
        overlay = frame.copy()
        for z in self.zones.values():
            x1, y1, x2, y2 = z.bbox
            cv2.rectangle(overlay, (x1, y1), (x2, y2), z.color, -1)
        cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)
        for z in self.zones.values():
            x1, y1, x2, y2 = z.bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), z.color, 2)
            cv2.putText(frame, z.name, (x1 + 6, y1 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, z.color, 1, cv2.LINE_AA)

    # ------------------------------------------------------------------
    # Mouse-draw fallback
    # ------------------------------------------------------------------

    def draw_zone_interactively(self, zone_name: str, frame: np.ndarray,
                                 win_name: str) -> tuple:
        """
        Block until user drags a rectangle for zone_name.
        Returns chosen (x1, y1, x2, y2) bbox.
        """
        result     = [None]
        start      = [None]
        cur_rect   = [None]

        def mouse_cb(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                start[0] = (x, y)
            elif event == cv2.EVENT_MOUSEMOVE and start[0]:
                cur_rect[0] = (*start[0], x, y)
            elif event == cv2.EVENT_LBUTTONUP and start[0]:
                x1 = min(start[0][0], x)
                y1 = min(start[0][1], y)
                x2 = max(start[0][0], x)
                y2 = max(start[0][1], y)
                result[0] = (x1, y1, x2, y2)
                start[0] = None

        cv2.setMouseCallback(win_name, mouse_cb)

        while result[0] is None:
            disp = frame.copy()
            _draw_instruction(disp, f"Draw zone '{zone_name}': click and drag, then release")
            if cur_rect[0]:
                x1, y1, x2, y2 = cur_rect[0]
                cv2.rectangle(disp, (x1, y1), (x2, y2),
                              self.COLORS.get(zone_name, (255, 255, 0)), 2)
            cv2.imshow(win_name, disp)
            k = cv2.waitKey(20) & 0xFF
            if k == ord('q'):
                raise SystemExit("Quit during zone setup.")

        cv2.setMouseCallback(win_name, lambda *a: None)
        return result[0]


# ---------------------------------------------------------------------------
# EmbeddingMatcher
# ---------------------------------------------------------------------------

class EmbeddingMatcher:
    """
    Stores one reference embedding per step and scores real-time vectors.

    Cosine similarity (default, fast)
    ----------------------------------
    Because PoseEstimator.normalize() produces a unit-scale vector (not
    unit-norm), we L2-normalise once more here so:
        cosine_sim(A, B) = dot(A_unit, B_unit)
    which collapses to a cheap NumPy dot product.

    Range: [-1, 1]; for valid hand poses you'll see [0.7, 1.0].
    Threshold 0.92 ≈ angular distance of ~23° in 42-D space.

    Procrustes similarity (optional)
    ---------------------------------
    scipy procrustes aligns two (21,2) point clouds by optimal rotation +
    uniform scale, then returns a residual disparity score ∈ [0, 1].
    Lower disparity = better match. We convert to similarity via:
        sim = max(0,  1 - disparity * 10)
    (the *10 factor empirically maps the ~0.0–0.1 range to 0–1).
    Useful when the hand may be rotated between capture and use.
    Slower (~0.5 ms) than cosine (~0.05 ms), so only use when needed.
    """

    def __init__(self):
        self.reference_vec: Optional[np.ndarray] = None  # unit 42-D
        self.reference_2d:  Optional[np.ndarray] = None  # (21,2) for Procrustes

    def set_reference(self, vector: np.ndarray,
                      landmarks_2d: Optional[np.ndarray] = None):
        norm = np.linalg.norm(vector)
        self.reference_vec = vector / norm if norm > 1e-9 else vector.copy()
        self.reference_2d  = landmarks_2d.copy() if landmarks_2d is not None else None

    @property
    def has_reference(self) -> bool:
        return self.reference_vec is not None

    def cosine_similarity(self, vector: np.ndarray) -> float:
        """Fast cosine sim. Returns 0.0 if no reference set."""
        if not self.has_reference:
            return 0.0
        norm = np.linalg.norm(vector)
        if norm < 1e-9:
            return 0.0
        return float(np.dot(self.reference_vec, vector / norm))

    def procrustes_similarity(self, landmarks_2d: np.ndarray) -> float:
        """Rotation-invariant Procrustes sim. Returns 0.0 if no reference."""
        if self.reference_2d is None:
            return 0.0
        try:
            _, _, disp = scipy_procrustes(self.reference_2d, landmarks_2d)
            return float(max(0.0, 1.0 - disp * 10))
        except Exception:
            return 0.0


# ---------------------------------------------------------------------------
# AssemblyMonitor  (orchestrator)
# ---------------------------------------------------------------------------

class AssemblyMonitor:
    """
    Orchestrates PoseEstimator, ZoneManager, EmbeddingMatcher instances,
    and drives a per-step state machine.

    State flow per step
    -------------------
    IDLE
     ↓  hand detected & enters item_zone  (or skipped if item_zone=None)
    IN_ITEM_ZONE
     ↓  hand moves into assembly_zone
    IN_ASSEMBLY_ZONE
     ↓  cosine_sim ≥ SIMILARITY_THRESHOLD  (and reference exists)
    POSE_MATCH
     ↓  hold for HOLD_FRAMES consecutive frames
    STEP_COMPLETE  →  auto-advance to next step
    """

    WIN = "Assembly Monitor"

    def __init__(self):
        self.pose  = PoseEstimator()
        self.zones = ZoneManager()
        self.matchers = [EmbeddingMatcher() for _ in STEPS]
        self.steps = STEPS

        self.current_step = 0
        self.state        = State.IDLE
        self.hold_count   = 0
        self.sim_score    = 0.0
        self.all_done     = False

        self._last_lm_px: Optional[np.ndarray] = None  # most recent landmarks

    # ------------------------------------------------------------------
    # Zone setup
    # ------------------------------------------------------------------

    def _setup_zones(self, cap):
        """Load pre-configured zones; interactively draw any that are None."""
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError("Cannot read camera frame during zone setup.")
        frame = cv2.flip(frame, 1)

        for name, bbox in ZONE_CONFIG.items():
            if bbox is not None:
                self.zones.add_zone(name, bbox)
            else:
                # Draw already-confirmed zones so user has context
                self.zones.draw_zones(frame)
                drawn = self.zones.draw_zone_interactively(name, frame, self.WIN)
                self.zones.add_zone(name, drawn)
                # Refresh frame for next zone
                ret, frame = cap.read()
                if ret:
                    frame = cv2.flip(frame, 1)

        print("[INFO] All zones configured:")
        for n, z in self.zones.zones.items():
            print(f"        {n}: {z.bbox}")

    # ------------------------------------------------------------------
    # Reference capture (R key)
    # ------------------------------------------------------------------

    def capture_reference(self):
        """Save current hand pose as reference for the current step."""
        if self._last_lm_px is None:
            print("[WARN] No hand in frame — cannot capture reference.")
            return
        vec = self.pose.normalize(self._last_lm_px)
        self.matchers[self.current_step].set_reference(vec, self._last_lm_px)
        print(f"[INFO] Reference captured — step {self.current_step}: "
              f"'{self.steps[self.current_step]['name']}'")

    # ------------------------------------------------------------------
    # State machine
    # ------------------------------------------------------------------

    def _tick(self, hand_center: Optional[tuple],
               norm_vector: Optional[np.ndarray]):
        """One state-machine tick per frame."""
        if self.all_done:
            return

        step    = self.steps[self.current_step]
        matcher = self.matchers[self.current_step]

        if hand_center is None:
            if self.state not in (State.STEP_COMPLETE,):
                self.state      = State.IDLE
                self.hold_count = 0
                self.sim_score  = 0.0
            return

        item_zone = step.get("item_zone")
        asm_zone  = step["assembly_zone"]

        in_item = (item_zone is None or
                   self.zones.point_in_zone(item_zone, hand_center))
        in_asm  = self.zones.point_in_zone(asm_zone, hand_center)

        self.sim_score = (matcher.cosine_similarity(norm_vector)
                          if norm_vector is not None else 0.0)

        match_ok = (matcher.has_reference and
                    self.sim_score >= SIMILARITY_THRESHOLD)

        # ---- transitions ----
        if self.state == State.IDLE:
            if item_zone is None and in_asm:
                self.state = State.IN_ASSEMBLY_ZONE
            elif in_item:
                self.state = State.IN_ITEM_ZONE

        elif self.state == State.IN_ITEM_ZONE:
            if in_asm:
                self.state = State.IN_ASSEMBLY_ZONE

        elif self.state == State.IN_ASSEMBLY_ZONE:
            if not in_asm:
                self.state = State.IDLE
            elif match_ok:
                self.state      = State.POSE_MATCH
                self.hold_count = 1

        elif self.state == State.POSE_MATCH:
            if not match_ok:
                self.state      = State.IN_ASSEMBLY_ZONE
                self.hold_count = 0
            else:
                self.hold_count += 1
                if self.hold_count >= HOLD_FRAMES:
                    self.state = State.STEP_COMPLETE

        elif self.state == State.STEP_COMPLETE:
            self._advance()

    def _advance(self):
        self.current_step += 1
        self.hold_count    = 0
        self.state         = State.IDLE
        self.sim_score     = 0.0
        if self.current_step >= len(self.steps):
            self.all_done     = True
            self.current_step = len(self.steps) - 1
            print("[INFO] *** All assembly steps complete! ***")
        else:
            print(f"[INFO] Step {self.current_step}: "
                  f"'{self.steps[self.current_step]['name']}'")

    # ------------------------------------------------------------------
    # Debug overlay
    # ------------------------------------------------------------------

    _STATE_COLOR = {
        State.IDLE:             (100, 100, 100),
        State.IN_ITEM_ZONE:     (30,  200, 255),
        State.IN_ASSEMBLY_ZONE: (0,   170, 230),
        State.POSE_MATCH:       (60,  230, 160),
        State.STEP_COMPLETE:    (60,  200,  80),
    }

    def draw_debug(self, frame: np.ndarray):
        """Overlay zones, landmarks, state badge, similarity bar, step bar."""
        h, w = frame.shape[:2]

        self.zones.draw_zones(frame)
        self.pose.draw_landmarks(frame)

        # ---- step progress bar (top) ----
        BAR_H = 38
        cv2.rectangle(frame, (0, 0), (w, BAR_H), (18, 18, 18), -1)
        sw = w // len(self.steps)
        for i, s in enumerate(self.steps):
            x0 = i * sw
            if i < self.current_step or (i == self.current_step and self.all_done):
                c = (60, 200, 80); lbl = f"v {s['name']}"
            elif i == self.current_step:
                c = (0, 200, 255); lbl = f"> {s['name']}"
            else:
                c = (70, 70, 70);  lbl = s['name']
            cv2.rectangle(frame, (x0 + 2, 2), (x0 + sw - 2, BAR_H - 2), c, 1)
            cv2.putText(frame, lbl, (x0 + 7, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, c, 1, cv2.LINE_AA)

        # ---- state badge ----
        sc = self._STATE_COLOR.get(self.state, (160, 160, 160))
        slabel = ("ALL DONE" if self.all_done
                  else self.state.name.replace("_", " "))
        cv2.rectangle(frame, (8, BAR_H + 6), (230, BAR_H + 32), sc, -1)
        cv2.putText(frame, slabel, (14, BAR_H + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (10, 10, 10), 1, cv2.LINE_AA)

        # ---- similarity bar ----
        matcher = self.matchers[self.current_step]
        by = BAR_H + 42
        if matcher.has_reference:
            BW = 200
            filled  = int(BW * max(0.0, self.sim_score))
            bar_c   = ((60, 230, 160) if self.sim_score >= SIMILARITY_THRESHOLD
                       else (50, 120, 220))
            cv2.rectangle(frame, (8, by), (8 + BW, by + 13), (45, 45, 45), -1)
            cv2.rectangle(frame, (8, by), (8 + filled, by + 13), bar_c, -1)
            cv2.putText(frame, f"sim {self.sim_score:.3f}", (8 + BW + 6, by + 11),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, bar_c, 1, cv2.LINE_AA)
            # hold-frame dots
            if self.state == State.POSE_MATCH:
                for d in range(HOLD_FRAMES):
                    dc = (60, 230, 160) if d < self.hold_count else (55, 55, 55)
                    cv2.circle(frame, (8 + d * 13 + 6, by + 26), 4, dc, -1)
        else:
            cv2.putText(frame, "No reference — press R to capture",
                        (8, by + 11),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 150, 255), 1, cv2.LINE_AA)

        # ---- hotkey legend (bottom) ----
        cv2.putText(frame, "R: capture ref   N: next step   Q: quit",
                    (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (140, 140, 140), 1, cv2.LINE_AA)

    # ------------------------------------------------------------------
    # Optional: post-exit part attachment check (stub)
    # ------------------------------------------------------------------

    def check_part_still_in_zone(self, frame: np.ndarray, zone_name: str,
                                  bg_roi: np.ndarray) -> bool:
        """
        [OPTIONAL] Detect whether a physical part remains in zone_name after
        the hand has left, using a stored background ROI.

        Algorithm:
        1. Capture bg_roi = zone ROI before the step (object present).
        2. After STEP_COMPLETE + hand leaves zone, capture current ROI.
        3. Compute mean absolute pixel diff between current and bg_roi.
        4. If diff < threshold → ROI looks like background → part was taken.
           If diff ≥ threshold → part still present → flag error.

        Returns True if part appears to have been picked up, False otherwise.
        Set PART_DIFF_THRESHOLD to ~15–25 depending on lighting conditions.
        """
        PART_DIFF_THRESHOLD = 20
        if zone_name not in self.zones.zones:
            return True
        x1, y1, x2, y2 = self.zones.zones[zone_name].bbox
        roi = frame[y1:y2, x1:x2]
        if roi.shape != bg_roi.shape:
            return True   # size mismatch, skip check
        diff = cv2.absdiff(roi, bg_roi)
        return float(diff.mean()) < PART_DIFF_THRESHOLD

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self):
        cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
        if not cap.isOpened():
            raise RuntimeError(f"Camera {CAMERA_INDEX} could not be opened.")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        cv2.namedWindow(self.WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WIN, 820, 620)

        print("[INFO] Setting up zones ...")
        self._setup_zones(cap)

        step_name = self.steps[self.current_step]['name']
        print(f"[INFO] Main loop started.  Step 0: '{step_name}'")
        print("[INFO] Hold the correct pose and press R to capture a reference.")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.03)
                    continue

                frame = cv2.flip(frame, 1)
                rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # --- Pose ---
                self.pose.process(rgb)
                lm_px            = self.pose.get_landmarks_px(frame.shape)
                self._last_lm_px = lm_px
                hand_center      = self.pose.get_hand_center_px(frame.shape)
                norm_vec         = (self.pose.normalize(lm_px)
                                    if lm_px is not None else None)

                # --- State tick ---
                self._tick(hand_center, norm_vec)

                # --- Overlay ---
                self.draw_debug(frame)
                cv2.imshow(self.WIN, frame)

                # --- Hotkeys ---
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.capture_reference()
                elif key == ord('n'):
                    print("[DEBUG] Force-advancing step.")
                    self._advance()

        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.pose.close()
            print("[INFO] Session ended.")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _draw_instruction(frame: np.ndarray, text: str):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, h - 52), (w, h), (18, 18, 18), -1)
    cv2.putText(frame, text, (12, h - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 220, 255), 2, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    monitor = AssemblyMonitor()
    monitor.run()
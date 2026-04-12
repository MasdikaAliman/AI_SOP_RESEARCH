"""
config.py — loads config.yaml and exposes typed, validated settings.
All other modules import from here; nothing reads the YAML directly.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import yaml


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class CameraConfig:
    source: str | int
    frame_w: int
    frame_h: int
    fps: int


@dataclass
class MediaPipeConfig:
    max_hands: int
    detection_confidence: float
    tracking_confidence: float
    model_complexity: int


@dataclass
class GestureConfig:
    grip_threshold: float
    success_delay: float
    pick_dwell_time: float


@dataclass
class ColorsConfig:
    white:  tuple
    green:  tuple
    red:    tuple
    yellow: tuple
    accent: tuple
    orange: tuple
    gray:   tuple
    purple: tuple

    def by_name(self, name: str) -> tuple:
        """Return a color tuple by string name (matches YAML clr_pick values)."""
        try:
            return getattr(self, name)
        except AttributeError:
            raise ValueError(f"Unknown color name '{name}'. "
                             f"Valid names: {list(self.__dataclass_fields__)}")


@dataclass
class InspectEncoderConfig:
    """
    Settings for the visual inspection encoder.
    Switch encoder via config.yaml inspect.encoder field.
    """
    encoder:              str   = "xfeat"     # "xfeat" | "lightglue"
    pass_threshold:       float = 20.0        # min inlier count to PASS
    max_num_keypoints:    int   = 2048        # XFeat top_k / LightGlue max_num_keypoints
    max_expected_inliers: int   = 50          # inlier count = similarity 1.0
    # XFeat only
    detection_threshold:  float = 0.05
    # LightGlue only
    lightglue_features:   str   = "aliked"   # aliked | superpoint | disk | sift
    depth_confidence:     float = 0.95
    width_confidence:     float = 0.99
    filter_threshold:     float = 0.1


# Backward-compat alias
DinoConfig = InspectEncoderConfig



@dataclass
class InspectConfig:
    """
    Per-step visual inspection settings.
    Only present on steps with mode == 'inspect'.

    crop_coords      : (y1, y2, x1, x2) on the resized display frame.
                       Matches the legacy step_crops format from main.py.
    reference_folder : path to folder of reference images (sorted = step order).
    pass_threshold   : cosine similarity required to PASS; falls back to
                       DinoConfig.pass_threshold when None.
    """
    crop_coords:      tuple            # (y1, y2, x1, x2)
    reference_folder: Path
    pass_threshold:   Optional[float]  # None → use global dino threshold


@dataclass
class SOPStep:
    step_id:     int
    name:        str
    instruction: str
    zone_pick:   tuple          # (x1, y1, x2, y2)
    clr_pick:    tuple          # BGR color tuple
    mode:        str = "hand_only"   # "hand_only" | "inspect"
    inspect:     Optional[InspectConfig] = None

    def __post_init__(self):
        if self.mode == "inspect" and self.inspect is None:
            raise ValueError(
                f"Step '{self.name}' (step_id={self.step_id}) has mode='inspect' "
                f"but no 'inspect' block defined in config.yaml."
            )
        if self.mode not in ("hand_only", "inspect"):
            raise ValueError(
                f"Step '{self.name}' has unknown mode '{self.mode}'. "
                f"Valid values: 'hand_only', 'inspect'."
            )

    @property
    def needs_inspect(self) -> bool:
        return self.mode == "inspect"


@dataclass
class AppConfig:
    camera:        CameraConfig
    mediapipe:     MediaPipeConfig
    gesture:       GestureConfig
    colors:        ColorsConfig
    dino:          DinoConfig
    assembly_zone: tuple           # (x1, y1, x2, y2)
    sop_steps:     list[SOPStep]

    # Derived — built once for O(1) wrong-zone lookups
    pick_zones: dict[int, tuple] = field(init=False)

    def __post_init__(self):
        self.pick_zones = {s.step_id: s.zone_pick for s in self.sop_steps}

    @property
    def has_inspect_steps(self) -> bool:
        """True if any step requires DINOv2 inspection — used to lazy-load model."""
        return any(s.needs_inspect for s in self.sop_steps)


# ── Loader ─────────────────────────────────────────────────────────────────────

def load_config(path: str | Path = "config.yaml") -> AppConfig:
    """
    Parse config.yaml and return a fully validated AppConfig.
    Raises ValueError with a clear message if anything is wrong.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path.resolve()}")

    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    # ── Colors (parsed first — steps reference by name) ───────────────────────
    raw_colors = raw["colors"]
    colors = ColorsConfig(**{k: tuple(v) for k, v in raw_colors.items()})

    # ── Camera ────────────────────────────────────────────────────────────────
    raw_cam = raw["camera"]
    source  = raw_cam["source"]
    try:
        source = int(source)
    except (ValueError, TypeError):
        pass

    camera = CameraConfig(
        source  = source,
        frame_w = int(raw_cam["frame_w"]),
        frame_h = int(raw_cam["frame_h"]),
        fps     = int(raw_cam["fps"]),
    )

    # ── MediaPipe ─────────────────────────────────────────────────────────────
    mp_raw    = raw["mediapipe"]
    mediapipe = MediaPipeConfig(
        max_hands            = int(mp_raw["max_hands"]),
        detection_confidence = float(mp_raw["detection_confidence"]),
        tracking_confidence  = float(mp_raw["tracking_confidence"]),
        model_complexity     = int(mp_raw["model_complexity"]),
    )

    # ── Gesture ───────────────────────────────────────────────────────────────
    g_raw   = raw["gesture"]
    gesture = GestureConfig(
        grip_threshold  = float(g_raw["grip_threshold"]),
        success_delay   = float(g_raw["success_delay"]),
        pick_dwell_time = float(g_raw["pick_dwell_time"]),
    )

    # -- XFeat / Inspect global config -------------------------------------------
    # -- Inspect encoder config (XFeat or LightGlue) ----------------------------
    raw_ins = raw.get("inspect", raw.get("dino", {}))  # support old "dino" key
    dino = InspectEncoderConfig(
        encoder              = str(raw_ins.get("encoder",              "xfeat")),
        pass_threshold       = float(raw_ins.get("pass_threshold",       20.0)),
        max_num_keypoints    = int(raw_ins.get("max_num_keypoints",    2048)),
        max_expected_inliers = int(raw_ins.get("max_expected_inliers",   50)),
        detection_threshold  = float(raw_ins.get("detection_threshold",  0.05)),
        lightglue_features   = str(raw_ins.get("lightglue_features",   "aliked")),
        depth_confidence     = float(raw_ins.get("depth_confidence",    0.95)),
        width_confidence     = float(raw_ins.get("width_confidence",    0.99)),
        filter_threshold     = float(raw_ins.get("filter_threshold",    0.1)),

    )

    # ── Assembly zone ─────────────────────────────────────────────────────────
    assembly_zone = tuple(raw["zones"]["assembly"])

    # ── SOP Steps ─────────────────────────────────────────────────────────────
    seen_ids: set[int] = set()
    sop_steps: list[SOPStep] = []

    for i, s in enumerate(raw["sop_steps"]):
        sid = int(s["step_id"])
        if sid in seen_ids:
            raise ValueError(f"Duplicate step_id {sid} in sop_steps[{i}]")
        seen_ids.add(sid)

        mode = str(s.get("mode", "hand_only"))

        # Parse inspect block if present
        inspect_cfg: Optional[InspectConfig] = None
        if "inspect" in s:
            raw_ins = s["inspect"]

            ref_folder = Path(raw_ins["reference_folder"])
            if not ref_folder.exists():
                import warnings
                warnings.warn(
                    f"Step '{s['name']}': reference_folder not found: "
                    f"{ref_folder.resolve()} — creating it now.",
                    UserWarning,
                )
                ref_folder.mkdir(parents=True, exist_ok=True)

            raw_crop = raw_ins["crop_coords"]
            if len(raw_crop) != 4:
                raise ValueError(
                    f"Step '{s['name']}': crop_coords must have 4 values "
                    f"[y1, y2, x1, x2], got {raw_crop}"
                )

            inspect_cfg = InspectConfig(
                crop_coords      = tuple(int(v) for v in raw_crop),
                reference_folder = ref_folder,
                pass_threshold   = (
                    float(raw_ins["pass_threshold"])
                    if "pass_threshold" in raw_ins
                    else None
                ),
            )

        sop_steps.append(SOPStep(
            step_id     = sid,
            name        = str(s["name"]),
            instruction = str(s["instruction"]),
            zone_pick   = tuple(s["zone_pick"]),
            clr_pick    = colors.by_name(s["clr_pick"]),
            mode        = mode,
            inspect     = inspect_cfg,
        ))

    # Sort by step_id to guarantee execution order
    sop_steps.sort(key=lambda s: s.step_id)

    return AppConfig(
        camera        = camera,
        mediapipe     = mediapipe,
        gesture       = gesture,
        colors        = colors,
        dino          = dino,
        assembly_zone = assembly_zone,
        sop_steps     = sop_steps,
    )
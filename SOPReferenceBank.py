"""
SOPReferenceBank.py — stores reference features for each SOP step.

Works with both encoders:
  - DINOv2Encoder : stores np.ndarray embeddings (N, 768)
  - XFeatEncoder  : stores list of feature dicts  [{keypoints, descriptors, scores}]

The encoder type is auto-detected from what encode() returns.
"""

from __future__ import annotations
import os
import json
import pickle
import numpy as np
from pathlib import Path
from PIL import Image


class SOPReferenceBank:
    """
    Offline step: register your SOP reference images once.

    Folder structure expected:
        data/sop_ref/step2/
            ref_01.jpg
            ref_02.jpg
            ...
    """

    def __init__(self, encoder):
        """
        encoder : XFeatEncoder or DINOv2Encoder instance.
        """
        self.encoder   = encoder
        self.steps     = []        # list of step metadata dicts
        # For DINOv2: np.ndarray [N, 768]
        # For XFeat : list of feature dicts
        self.embeddings = None
        self._mode      = None     # 'dino' | 'xfeat'

    # ── Registration ──────────────────────────────────────────────────────────

    def register_from_folder(self, folder_path: str):
        """Load all images sorted by filename as SOP reference images."""
        files = sorted([
            f for f in os.listdir(folder_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])

        if not files:
            raise ValueError(f"No images found in '{folder_path}'")

        images, metadata = [], []
        for idx, fname in enumerate(files):
            img = Image.open(os.path.join(folder_path, fname)).convert('RGB')
            images.append(img)
            metadata.append({
                'step_id':   idx,
                'step_name': os.path.splitext(fname)[0],
                'filename':  fname,
            })
            print(f"  Registered ref {idx}: {fname}")

        print(f"\n  Encoding {len(images)} reference image(s)...")
        self._encode_and_store(images, metadata)
        print(f"  Reference bank ready: {len(self.steps)} step(s)")

    def register_single(self, image, step_name: str):
        """Register one reference image manually."""
        feat = self.encoder.encode(image)
        meta = {'step_id': len(self.steps), 'step_name': step_name}

        if self._is_xfeat_feat(feat):
            self._mode = 'xfeat'
            if self.embeddings is None:
                self.embeddings = [feat]
            else:
                self.embeddings.append(feat)
        else:
            self._mode = 'dino'
            emb = feat
            if self.embeddings is None:
                self.embeddings = emb[np.newaxis, :]
            else:
                self.embeddings = np.vstack([self.embeddings, emb])

        self.steps.append(meta)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str):
        """
        Save reference bank to disk.
        DINOv2: saves .npy + .json
        XFeat : saves .pkl + .json  (feature dicts not easily npy-able)
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        if self._mode in ('xfeat', 'lightglue'):
            # Local feature dicts — save as pickle
            with open(path + '_features.pkl', 'wb') as f:
                pickle.dump(self.embeddings, f)
        else:
            # DINOv2 embeddings — save as numpy
            np.save(path + '_embeddings.npy', self.embeddings)

        with open(path + '_metadata.json', 'w') as f:
            json.dump({'mode': self._mode, 'steps': self.steps}, f, indent=2)

        print(f"  [Bank] Saved to '{path}' (mode={self._mode})")

    def load(self, path: str):
        """Load reference bank from disk."""
        meta_path = path + '_metadata.json'
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata not found: {meta_path}")

        with open(meta_path) as f:
            meta = json.load(f)

        self._mode = meta.get('mode', 'dino')
        self.steps = meta['steps']

        if self._mode == 'xfeat':
            pkl_path = path + '_features.pkl'
            if not os.path.exists(pkl_path):
                raise FileNotFoundError(f"Feature file not found: {pkl_path}")
            with open(pkl_path, 'rb') as f:
                self.embeddings = pickle.load(f)
        else:
            npy_path = path + '_embeddings.npy'
            if not os.path.exists(npy_path):
                raise FileNotFoundError(f"Embeddings not found: {npy_path}")
            self.embeddings = np.load(npy_path)

        print(f"  [Bank] Loaded {len(self.steps)} step(s) (mode={self._mode})")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _encode_and_store(self, images: list, metadata: list):
        sample = self.encoder.encode(images[0])

        if self._is_local_feat(sample):
            # XFeat or LightGlue — both return feature dicts with keypoints.
            # Detect which by checking encoder class name.
            enc_name = type(self.encoder).__name__.lower()
            self._mode = 'lightglue' if 'lightglue' in enc_name else 'xfeat'
            self.embeddings = [sample] + [self.encoder.encode(img) for img in images[1:]]
        else:
            self._mode      = 'dino'
            # DINOv2 supports batch encoding
            if hasattr(self.encoder, 'encode_batch'):
                self.embeddings = self.encoder.encode_batch(images)
            else:
                self.embeddings = np.stack([self.encoder.encode(img) for img in images])

        self.steps = metadata
        print(f"  [Bank] Encoder mode detected: {self._mode}")

    @staticmethod
    def _is_local_feat(feat) -> bool:
        """True if the encoder returned a local feature dict (XFeat or LightGlue)."""
        return isinstance(feat, dict) and 'keypoints' in feat

    @staticmethod
    def _is_xfeat_feat(feat) -> bool:
        """Backward compat alias."""
        return SOPReferenceBank._is_local_feat(feat)
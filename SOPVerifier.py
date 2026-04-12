"""
SOPVerifier.py — verifies a live frame against SOP reference images.

Works with both encoders:
  - DINOv2Encoder : compares cosine similarity of global embeddings
  - XFeatEncoder  : compares inlier count of local feature matches

The pass_threshold meaning changes per encoder:
  DINOv2 : cosine similarity  0.0–1.0  (e.g. 0.82)
  XFeat  : inlier count       0–top_k  (e.g. 15–40 inliers)

Set pass_threshold accordingly in config.yaml per step.
"""

from __future__ import annotations
import numpy as np
from SOPReferenceBank import SOPReferenceBank
from icecream import ic

class SOPVerifier:

    def __init__(self, encoder, bank: SOPReferenceBank,
                 pass_threshold: float = 20.0):
        """
        encoder        : XFeatEncoder or DINOv2Encoder
        bank           : SOPReferenceBank (already loaded/registered)
        pass_threshold : inlier count for XFeat (e.g. 20)
                         cosine similarity for DINOv2 (e.g. 0.82)
        """
        self.encoder   = encoder
        self.bank      = bank
        self.threshold = pass_threshold
        self.current_step = 0
        self.history      = []
        self._mode        = bank._mode   # 'xfeat' | 'dino'

    # ── Main entry ────────────────────────────────────────────────────────────

    def verify(self, frame) -> dict:
        """
        Verify a live frame against the expected SOP step.

        frame : PIL.Image or np.ndarray (OpenCV BGR)

        Returns a result dict compatible with sop_engine.py:
          {
            'expected_step'   : int
            'expected_name'   : str
            'similarity'      : float  (inliers or cosine sim)
            'passed'          : bool
            'best_match_step' : int
            'best_match_name' : str
            'best_match_sim'  : float
            'message'         : str
            # XFeat only:
            'inliers'         : int
            'total_matches'   : int
            'mkpts_ref'       : np.ndarray (M, 2)
            'mkpts_live'      : np.ndarray (M, 2)
          }
        """
        if self._mode in ('xfeat', 'lightglue'):
            # Both XFeat and LightGlue use encoder.match() — same code path
            result = self._verify_xfeat(frame)
        else:
            result = self._verify_dino(frame)

        self.history.append(result)
        return result

    # ── XFeat verification ────────────────────────────────────────────────────

    def _verify_xfeat(self, frame) -> dict:
        feat_live = self.encoder.encode(frame)
        expected  = self.current_step
        ref_feats = self.bank.embeddings   # list of feature dicts

        # Match against expected step
        match_expected = self.encoder.match(ref_feats[expected], feat_live)
        inliers_expected = match_expected['inliers']
        ic(inliers_expected, match_expected['total_matches'], match_expected['similarity'], self.threshold)
        passed = match_expected["similarity"] >= self.threshold

        # Find best match across all steps (for wrong-step hints)
        best_step, best_inliers = expected, inliers_expected
        for i, ref_feat in enumerate(ref_feats):
            if i == expected:
                continue
            m = self.encoder.match(ref_feat, feat_live)
            if m['inliers'] > best_inliers:
                best_inliers = m['inliers']
                best_step    = i

        result = {
            'expected_step':   expected,
            'expected_name':   self.bank.steps[expected]['step_name'],
            'similarity':      float(match_expected["similarity"]),
            'passed':          passed,
            'best_match_step': best_step,
            'best_match_name': self.bank.steps[best_step]['step_name'],
            'best_match_sim':  float(best_inliers),
            'inliers':         inliers_expected,
            'total_matches':   match_expected['total_matches'],
            'mkpts_ref':       match_expected['mkpts_ref'],
            'mkpts_live':      match_expected['mkpts_live'],
        }

        if passed:
            result['message'] = (
                f"PASS — {result['expected_name']} "
                f"({inliers_expected} inliers ≥ {self.threshold:.0f})"
            )
        else:
            if best_step != expected and best_inliers >= self.threshold:
                result['message'] = (
                    f"WRONG STEP — detected '{result['best_match_name']}' "
                    f"but expected '{result['expected_name']}'"
                )
            else:
                result['message'] = (
                    f"NOT RECOGNISED — {inliers_expected} inliers "
                    f"(need ≥ {self.threshold:.0f})"
                )

        return result

    # ── DINOv2 verification (original logic, unchanged) ───────────────────────

    def _verify_dino(self, frame) -> dict:
        z_live   = self.encoder.encode(frame)
        expected = self.current_step
        z_ref    = self.bank.embeddings[expected]

        sim_expected = float(np.dot(z_live, z_ref))
        all_sims     = self.bank.embeddings @ z_live
        best_step    = int(np.argmax(all_sims))
        best_sim     = float(all_sims[best_step])
        passed       = sim_expected >= self.threshold

        result = {
            'expected_step':   expected,
            'expected_name':   self.bank.steps[expected]['step_name'],
            'similarity':      round(sim_expected, 4),
            'passed':          passed,
            'best_match_step': best_step,
            'best_match_name': self.bank.steps[best_step]['step_name'],
            'best_match_sim':  round(best_sim, 4),
            'all_similarities': {
                s['step_name']: round(float(all_sims[i]), 4)
                for i, s in enumerate(self.bank.steps)
            },
        }

        if passed:
            result['message'] = f"PASS — step {expected} confirmed"
        else:
            if best_step != expected and best_sim >= self.threshold:
                result['message'] = (
                    f"WRONG STEP — you are doing '{result['best_match_name']}' "
                    f"but expected '{result['expected_name']}'"
                )
            else:
                result['message'] = (
                    f"NOT RECOGNISED — similarity {sim_expected:.2f} "
                    f"< threshold {self.threshold}"
                )

        return result

    # ── Utility ───────────────────────────────────────────────────────────────

    def reset(self):
        self.current_step = 0
        self.history      = []

    def jump_to_step(self, step_id: int):
        self.current_step = step_id
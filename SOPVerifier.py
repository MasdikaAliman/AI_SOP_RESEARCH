from DINOv2Encoder import  DINOv2Encoder
from SOPReferenceBank import SOPReferenceBank
import numpy as np

class SOPVerifier:
    def __init__(self, encoder: DINOv2Encoder, bank: SOPReferenceBank,
                 pass_threshold: float = 0.82):
        self.encoder = encoder
        self.bank = bank
        self.threshold = pass_threshold
        self.current_step = 0  # which step we expect next
        self.history = []

    def verify(self, frame) -> dict:
        """
        Main entry point. Call on every captured frame.
        frame: PIL.Image or np.ndarray (OpenCV BGR)

        Returns a result dict with status, similarity, and guidance.
        """
        z_live = self.encoder.encode(frame)
        expected_step = self.current_step

        # Similarity vs the EXPECTED step
        z_ref = self.bank.embeddings[expected_step]
        sim_expected = float(np.dot(z_live, z_ref))  # already L2-normed

        # Find the closest step overall (for wrong-step hints)
        # all_sims = self.bank.embeddings @ z_live  # [N,] dot products
        # best_step_idx = int(np.argmax(all_sims))
        # best_sim = float(all_sims[best_step_idx])

        passed = sim_expected >= self.threshold

        result = {
            # 'expected_step': expected_step,
            'expected_name': self.bank.steps[expected_step]['step_name'],
            'similarity': round(sim_expected, 4),
            'passed': passed,
            # 'best_match_step': best_step_idx,
            # 'best_match_name': self.bank.steps[best_step_idx]['step_name'],
            # 'best_match_sim': round(best_sim, 4),
            # 'all_similarities': {
            #     s['step_name']: round(float(all_sims[i]), 4)
            #     for i, s in enumerate(self.bank.steps)
            # }
        }

        if passed:
            result['message'] = f"PASS — step {expected_step} confirmed"
            # self._advance_step()
        else:
            # if best_step_idx != expected_step and best_sim >= self.threshold:
            #     result['message'] = (
            #         f"WRONG STEP — you are doing '{result['best_match_name']}' "
            #         f"but expected '{result['expected_name']}'"
            #     )
            # else:
                result['message'] = (
                    f"NOT RECOGNIZED — similarity {sim_expected:.2f} "
                    f"< threshold {self.threshold}"
                )

        # self.history.append(result)
        return result

    def _advance_step(self):
        if self.current_step < len(self.bank.steps) - 1:
            self.current_step += 1
        else:
            self.current_step = 0  # loop / reset

    def reset(self):
        self.current_step = 0
        self.history = []

    def jump_to_step(self, step_id: int):
        """Manually set which step to expect — useful for resuming."""
        self.current_step = step_id
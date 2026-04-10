import json

import cv2
import icecream
import numpy as np
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pickle  # To save/load processed reference features


class FeatureBasedVerifier:
    def __init__(self,
                 feature_extractor: str = 'SIFT',  # or 'ORB'
                 matcher_type: str = 'FLANN',  # or 'BF'
                 match_ratio_threshold: float = 0.75,  # Lowe's ratio test
                 min_matches_needed: int = 10,
                 pass_threshold: float = 0.6,  # Fraction of min_matches_needed
                 reference_folders: Dict[int, Path] = None,  # {step_id: folder_path}
                 cache_dir: Path = Path("feature_cache")):

        self.feature_extractor_name = feature_extractor
        self.matcher_type = matcher_type
        self.match_ratio_threshold = match_ratio_threshold
        self.min_matches_needed = min_matches_needed
        self.pass_threshold = pass_threshold
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True)  # Create cache directory if it doesn't exist

        # Initialize feature detector and descriptor
        if feature_extractor.upper() == 'SIFT':
            self.detector = cv2.SIFT_create()
        elif feature_extractor.upper() == 'ORB':
            self.detector = cv2.ORB_create()
        else:
            raise ValueError(f"Unsupported feature extractor: {feature_extractor}")

        # Initialize matcher
        if matcher_type.upper() == 'FLANN':
            # FLANN parameters might need tuning based on descriptor type (SIFT/ORB)
            if feature_extractor.upper() == 'SIFT':
                FLANN_INDEX_KDTREE = 1
                index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            elif feature_extractor.upper() == 'ORB':
                FLANN_INDEX_LSH = 6
                index_params = dict(algorithm=FLANN_INDEX_LSH,
                                    table_number=6,  # 12
                                    key_size=12,  # 20
                                    multi_probe_level=1)  # 2
            search_params = dict(checks=50)  # or pass empty {}
            self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        elif matcher_type.upper() == 'BF':
            # Use normType based on descriptor type (L2 for SIFT, HAMMING for ORB)
            norm_type = cv2.NORM_L2 if feature_extractor.upper() == 'SIFT' else cv2.NORM_HAMMING
            self.matcher = cv2.BFMatcher(norm_type, crossCheck=False)  # Cross-check often not used with ratio test
        else:
            raise ValueError(f"Unsupported matcher type: {matcher_type}")

        # Dictionary to store reference features: {step_id: {'kp_list': [...], 'desc_list': [...]}}
        self.ref_features: Dict[int, Dict[str, List]] = {}
        self.load_reference_features(reference_folders)

    def _keypoints_to_dict(self, kp_list: List[cv2.KeyPoint]) -> List[dict]:
        """Convert KeyPoint objects to a list of dictionaries."""
        return [
            {
                'pt': kp.pt,
                'angle': kp.angle,
                'class_id': kp.class_id,
                'octave': kp.octave,
                'response': kp.response,
                'size': kp.size
            }
            for kp in kp_list
        ]

    def _dict_to_keypoints(self, kp_dict_list: List[dict]) -> List[cv2.KeyPoint]:
        """Convert a list of dictionaries back to KeyPoint objects."""
        return [
            cv2.KeyPoint(
                x=kp_dict['pt'][0],
                y=kp_dict['pt'][1],
                _size=kp_dict['size'],  # Note: '_size' maps to 'size' property
                _angle=kp_dict['angle'],
                _response=kp_dict['response'],
                _octave=kp_dict['octave'],
                _class_id=kp_dict['class_id']
            )
            for kp_dict in kp_dict_list
        ]

    def _load_image_features(self, image_path: Path):
        """Load and compute features for a single image."""
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)  # Load as grayscale
        if img is None:
            print(f"[ERROR] Could not load image: {image_path}")
            return [], np.array([])  # Return empty lists if image fails to load

        kp, desc = self.detector.detectAndCompute(img, None)
        if desc is None:  # Some descriptors (like ORB) might return None if no keypoints found
            desc = np.array([]).reshape(-1, self.detector.descriptorSize())
        return kp, desc

    def _process_reference_folder(self, folder_path: Path):
        """Process all images in a reference folder to extract features."""
        print(f"[INFO] Processing reference folder: {folder_path}")
        all_kp = []
        all_desc = []
        num_images = 0

        for img_file in folder_path.glob("*"):  # Iterate through all files in folder
            if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                kp, desc = self._load_image_features(img_file)
                if len(kp) > 0 and desc.size > 0:  # Only add if features were found
                    all_kp.extend(kp)
                    if len(all_desc) == 0:
                        all_desc = desc
                    else:
                        all_desc = np.vstack((all_desc, desc))
                    num_images += 1
                else:
                    print(f"[WARN] No features found in image: {img_file}")

        if len(all_desc) == 0:
            print(f"[ERROR] No features found in any image in folder: {folder_path}")
            return [], np.array([]).reshape(-1, self.detector.descriptorSize())

        print(f"[INFO] Loaded features from {num_images} images in {folder_path}")
        return all_kp, all_desc

    def load_reference_features(self, reference_folders: Optional[Dict[int, Path]]):
        """Load or compute and cache features for all reference folders."""
        if not reference_folders:
            print("[WARN] No reference folders provided to FeatureBasedVerifier.")
            return

        for step_id, folder_path in reference_folders.items():
            # Use .json for keypoints, .npy for descriptors
            cache_kp_file = self.cache_dir / f"features_step_{step_id}_{self.feature_extractor_name}_{self.matcher_type}_kp.json"
            cache_desc_file = self.cache_dir / f"features_step_{step_id}_{self.feature_extractor_name}_{self.matcher_type}_desc.npy"

            if cache_kp_file.exists() and cache_desc_file.exists():
                print(f"[CACHE] Loading cached features for step {step_id} from {cache_kp_file} and {cache_desc_file}")
                try:
                    # Load keypoints
                    with open(cache_kp_file, 'r') as f:
                        cached_kp_dicts = json.load(f)
                    cached_kp_list = self._dict_to_keypoints(cached_kp_dicts)

                    # Load descriptors
                    cached_desc_array = np.load(cache_desc_file, allow_pickle=False)

                    self.ref_features[step_id] = {
                        'kp_list': cached_kp_list,
                        'desc_array': cached_desc_array
                    }
                    print(f"[CACHE] Successfully loaded cached features for step {step_id}")
                except Exception as e:
                    print(f"[ERROR] Failed to load cache for step {step_id}: {e}. Recomputing...")
                    # Fall through to recompute
                    kp_list, desc_array = self._process_reference_folder(folder_path)
                    self._save_features_to_cache(step_id, kp_list, desc_array)
            else:
                print(f"[PROCESS] Computing features for step {step_id} in {folder_path}")
                kp_list, desc_array = self._process_reference_folder(folder_path)
                self._save_features_to_cache(step_id, kp_list, desc_array)

            if self.ref_features[step_id]['desc_array'].size == 0:
                print(f"[ERROR] Could not compute features for step {step_id}, skipping.")
                # Assign empty arrays if computation failed
                self.ref_features[step_id] = {'kp_list': [],
                                              'desc_array': np.array([]).reshape(-1, self.detector.descriptorSize())}

    def _save_features_to_cache(self, step_id: int, kp_list: List[cv2.KeyPoint], desc_array: np.ndarray):
        """Helper to save features to cache files."""
        if len(desc_array) > 0:
            cache_kp_file = self.cache_dir / f"features_step_{step_id}_{self.feature_extractor_name}_{self.matcher_type}_kp.json"
            cache_desc_file = self.cache_dir / f"features_step_{step_id}_{self.feature_extractor_name}_{self.matcher_type}_desc.npy"

            # Save keypoints
            kp_dict_list = self._keypoints_to_dict(kp_list)
            with open(cache_kp_file, 'w') as f:
                json.dump(kp_dict_list, f)

            # Save descriptors
            np.save(cache_desc_file, desc_array, allow_pickle=False)

            print(f"[CACHE] Cached features for step {step_id} to {cache_kp_file} and {cache_desc_file}")
            self.ref_features[step_id] = {
                'kp_list': kp_list,
                'desc_array': desc_array
            }
        else:
            print(f"[ERROR] Cannot save empty features for step {step_id}")

    def verify(self, frame: np.ndarray, expected_step_id: int) -> dict:
        """
        Main verification function.
        Args:
            frame: The live camera frame (BGR).
            expected_step_id: The step ID for which the item is expected.
        Returns:
            A dictionary containing the verification result.
        """
        if expected_step_id not in self.ref_features:
            print(f"[ERROR] No reference features loaded for step_id {expected_step_id}")
            return {
                'expected_step': expected_step_id,
                'passed': False,
                'message': f"No reference data for step {expected_step_id}",
                'similarity': 0.0  # Using 'similarity' key for consistency, though meaning differs
            }

        ref_desc = self.ref_features[expected_step_id]['desc_array']

        # --- Feature Matching ---
        # Convert frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect and compute features on the live frame
        live_kp, live_desc = self.detector.detectAndCompute(gray_frame, None)

        if live_desc is None or live_desc.size == 0 or ref_desc.size == 0:
            # print(f"[DEBUG] No descriptors found in live frame or reference for step {expected_step_id}") # Uncomment for debugging
            return {
                'expected_step': expected_step_id,
                'passed': False,
                'message': "No features detected in live frame or reference",
                'similarity': 0.0
            }

        # Match features
        matches = self.matcher.knnMatch(live_desc, ref_desc, k=2)  # Get top 2 matches for each live descriptor

        # Apply Lowe's ratio test to filter good matches
        good_matches = []
        if matches:
            for m_n in matches:
                if len(m_n) == 2:  # Ensure both matches exist
                    m, n = m_n
                    if m.distance < self.match_ratio_threshold * n.distance:
                        good_matches.append(m)

        num_good_matches = len(good_matches)

        total_ref_features = len(self.ref_features[expected_step_id]['kp_list'])

        # Calculate a "similarity" score based on good matches
        # This is a heuristic and might need adjustment
        good_matches_ratio = num_good_matches / max(total_ref_features, 1)  # Avoid division by zero
        similarity_score = good_matches_ratio  # Or calculate differently if needed

        # Determine pass/fail
        required_matches = int(self.pass_threshold * self.min_matches_needed)
        passed = num_good_matches >= required_matches  # Or use good_matches_ratio >= self.pass_threshold
        icecream.ic(num_good_matches, total_ref_features, required_matches)
        result = {
            'expected_step': expected_step_id,
            'passed': passed,
            'similarity': round(similarity_score, 4),  # Consistent key name
            'message': f"Found {num_good_matches}/{required_matches} required matches" if passed else f"Only {num_good_matches}/{required_matches} matches found"
        }

        # Optional: Print debug info
        # print(f"[DEBUG] Step {expected_step_id}: Matches: {num_good_matches}, Ratio: {good_matches_ratio:.2f}, Passed: {passed}")

        return result



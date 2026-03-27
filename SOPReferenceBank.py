import os
import json
from DINOv2Encoder import DINOv2Encoder
import numpy as np
from PIL import Image
class SOPReferenceBank:
    """
    Offline step: register your SOP images once.
    Folder structure expected:
        sop_images/
            step_01_pick_component.jpg
            step_02_insert_bolt.jpg
            step_03_torque_wrench.jpg
            ...
    """

    def __init__(self, encoder: DINOv2Encoder):
        self.encoder = encoder
        self.steps = []  # list of step metadata dicts
        self.embeddings = None  # np.ndarray [N, 768]

    def register_from_folder(self, folder_path: str):
        """Load all images sorted by filename as SOP steps."""
        files = sorted([
            f for f in os.listdir(folder_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])

        images, metadata = [], []
        for idx, fname in enumerate(files):
            img = Image.open(os.path.join(folder_path, fname)).convert('RGB')
            images.append(img)
            metadata.append({
                'step_id': idx,
                'step_name': os.path.splitext(fname)[0],
                'filename': fname
            })
            print(f"Registered step {idx}: {fname}")

        print(f"\nEncoding {len(images)} SOP steps with DINOv2...")
        self.embeddings = self.encoder.encode_batch(images)
        self.steps = metadata
        print(f"Reference bank ready: {self.embeddings.shape}")

    def register_single(self, image, step_name: str):
        """Register one step manually."""
        emb = self.encoder.encode(image)
        if self.embeddings is None:
            self.embeddings = emb[np.newaxis, :]
        else:
            self.embeddings = np.vstack([self.embeddings, emb])
        self.steps.append({
            'step_id': len(self.steps),
            'step_name': step_name,
        })

    def save(self, path: str):
        np.save(path + '_embeddings.npy', self.embeddings)
        with open(path + '_metadata.json', 'w') as f:
            json.dump(self.steps, f, indent=2)
        print(f"Saved reference bank to {path}")

    def load(self, path: str):
        self.embeddings = np.load(path + '_embeddings.npy')
        with open(path + '_metadata.json') as f:
            self.steps = json.load(f)
        print(f"Loaded {len(self.steps)} SOP steps")
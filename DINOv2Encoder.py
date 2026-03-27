import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np


class DINOv2Encoder:
    def __init__(self, model_name='dinov2_vitb14', device=None):
        self.device = device or ('cpu' if torch.cuda.is_available() else 'cpu')
        # Load DINOv2 from torch hub — no training needed
        self.model = torch.hub.load('facebookresearch/dinov2', model_name)
        self.model.eval().to(self.device)

        # DINOv2 official preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    @torch.no_grad()
    def encode(self, image) -> np.ndarray:
        """
        image: PIL.Image or np.ndarray (BGR from OpenCV)
        returns: L2-normalized 768-dim embedding
        """
        if isinstance(image, np.ndarray):
            # OpenCV BGR → PIL RGB
            image = Image.fromarray(image[:, :, ::-1])

        tensor = self.transform(image).unsqueeze(0).to(self.device)
        embedding = self.model(tensor)  # [1, 768]
        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding.cpu().numpy()[0]  # (768,)

    @torch.no_grad()
    def encode_batch(self, images: list) -> np.ndarray:
        tensors = torch.stack([
            self.transform(img if isinstance(img, Image.Image)
                           else Image.fromarray(img[:, :, ::-1]))
            for img in images
        ]).to(self.device)
        embeddings = self.model(tensors)
        return F.normalize(embeddings, p=2, dim=1).cpu().numpy()
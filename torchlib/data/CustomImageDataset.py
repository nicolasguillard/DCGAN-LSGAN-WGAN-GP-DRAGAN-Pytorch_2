import os
from typing import Callable, Optional
import torch
from torch.utils.data import Dataset
from PIL import Image


class CustomImageDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, images_dir: str, transforms: Optional[Callable] = None, extension: str = '.jpg'):
        """
        Arguments:
            images_dir (string): Directory with all the images.
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.img_dir = images_dir
        self.transform = transforms
        self.extension = extension
        self.images = [f for f in os.listdir(self.img_dir) if f.lower().endswith(self.extension)]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.img_dir, self.images[idx])
        image = CustomImageDataset.pil_loader(img_path)
        if self.transform:
            image = self.transform(image)
        return image

    @staticmethod
    def pil_loader(path):
        with open(path, 'rb') as f:
            return Image.open(f).convert('RGB')

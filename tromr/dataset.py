import os  # Ensure the os module is imported
import torch
from torch.utils.data import Dataset
from PIL import Image
import json

class CustomDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, transform=None):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.transform = transform
        self.image_files = sorted(os.listdir(image_dir))  # List image files
        self.annotation_files = sorted(os.listdir(annotation_dir))  # List annotation files

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        annotation_path = os.path.join(self.annotation_dir, self.annotation_files[idx])

        # Load image
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # Load annotations
        with open(annotation_path, 'r') as f:
            annotations = json.load(f)

        # Return image and annotations
        return {'image': image, 'annotations': torch.tensor(annotations['symbols'])}


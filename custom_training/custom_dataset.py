"""
Dataset and augmentation utilities for self-supervised learning.
"""

import torch
import torch.utils.data
import torchvision.transforms as transforms
from glob import glob
from natsort import natsorted
from PIL import Image

class RandomResizedCrop:
    """Random resized crop augmentation."""

    def __init__(self, size, scale=(0.2, 1.0)):
        self.size = size
        self.scale = scale

    def __call__(self, img):
        return transforms.RandomResizedCrop(self.size, scale=self.scale)(img)


class ColorJitter:
    """Color jitter augmentation."""

    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, prob=0.8):
        self.transform = transforms.ColorJitter(brightness, contrast, saturation, hue)
        self.prob = prob

    def __call__(self, img):
        if torch.rand(1) < self.prob:
            return self.transform(img)
        return img


class Grayscale:
    """Grayscale augmentation."""

    def __init__(self, prob=0.2):
        self.prob = prob

    def __call__(self, img):
        if torch.rand(1) < self.prob:
            return transforms.Grayscale(num_output_channels=3)(img)
        return img


class Solarization:
    """Solarization augmentation."""

    def __init__(self, prob=0.1):
        self.prob = prob

    def __call__(self, img):
        if torch.rand(1) < self.prob:
            img = transforms.functional.solarize(img, threshold=128)
        return img


class HorizontalFlip:
    """Horizontal flip augmentation."""

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img):
        if torch.rand(1) < self.prob:
            return transforms.functional.hflip(img)
        return img


def get_train_transforms(image_size):
    """Get training transforms for self-supervised learning."""
    transform = transforms.Compose(
        [
            RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, prob=0.8
            ),
            Grayscale(prob=0.2),
            Solarization(prob=0.1),
            HorizontalFlip(prob=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.5368, 0.5368, 0.5368), (0.1977, 0.1977, 0.1977)),
        ]
    )

    return transform


def get_val_transforms():
    """Get validation transforms."""
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5368, 0.5368, 0.5368), (0.1977, 0.1977, 0.1977)),
        ]
    )


class ImageDataset(torch.utils.data.Dataset):
    """Custom dataset that applies augmentations multiple times to create views.

    Returns only the list of views: SSL training needs no labels, and the
    (PIL) mask would not match the randomly cropped views anyway.
    """

    def __init__(self, dataset, transform, num_crops=2):
        self.dataset = dataset
        self.transform = transform
        self.num_crops = num_crops

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, _ = self.dataset[idx]
        return [self.transform(image) for _ in range(self.num_crops)]


class RockDataset(torch.utils.data.Dataset):
    def __init__(self, root, train=True, ending:str = ".png"):
        self.img_paths = natsorted(glob(f"{root}/imgs/{"train" if train else "val"}/*{ending}"))
        self.mask_paths = natsorted(glob(f"{root}/masks/{"train" if train else "val"}/*{ending}"))
    
    def __len__(self):
        if len(self.img_paths) == len(self.mask_paths):
            return len(self.img_paths)
        else:
            raise Exception(f"image path length ({len(self.img_paths)}) and mask path length ({len(self.mask_paths)}) do not have the same length")

    def __getitem__(self, idx):
        image = Image.open(self.img_paths[idx]).convert("RGB")
        label = Image.open(self.mask_paths[idx]).convert("L")
        return image, label
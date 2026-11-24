"""
Evaluation utilities for self-supervised learning on the RockDataset (segmentation).

RockDataset yields (PIL RGB image, PIL "L" mask) pairs, so representations are evaluated
with a *segmentation* linear probe: a 1x1 conv on the backbone's spatial feature map,
upsampled to the mask resolution and trained/evaluated with per-pixel cross-entropy.

Interface kept compatible with main.py:
    LinearProbe(feature_dim, num_classes)
    evaluate_linear_probe(model, linear_probe, val_loader, device, use_amp) -> (pixel_acc, loss)

Typical usage:
    val_dataset = SegmentationEvalDataset(RockDataset(root, train=False), num_classes=2)
    val_loader = DataLoader(val_dataset, ...)
    probe = LinearProbe(features_dim, num_classes=2).to(device)
    train_linear_probe(model, probe, probe_train_loader, device)
    metrics = evaluate_segmentation_probe(model, probe, val_loader, device, num_classes=2)
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.amp import autocast
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from tqdm import tqdm

from custom_dataset import get_val_transforms

IGNORE_INDEX = 255


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def mask_to_target(mask, num_classes):
    """Convert a PIL "L" mask to a LongTensor of class indices (H, W).

    - num_classes == 2: binary mask, every pixel > 127 is foreground (1).
    - otherwise: pixel values are used as class ids; values >= num_classes are ignored.
    """
    m = torch.from_numpy(np.array(mask, dtype=np.int64))
    if num_classes == 2:
        return (m > 127).long()
    m[m >= num_classes] = IGNORE_INDEX
    return m


class SegmentationEvalDataset(torch.utils.data.Dataset):
    """Wraps RockDataset: resizes image and mask jointly and converts them to tensors.

    Image -> resized (bilinear) + get_val_transforms() (ToTensor + Normalize).
    Mask  -> resized (nearest) + mask_to_target().
    image_size must match the training resolution (ViT backbones require exactly 32).
    """

    def __init__(self, dataset, num_classes=2, image_size=32):
        self.dataset = dataset
        self.num_classes = num_classes
        self.image_size = image_size
        self.image_transform = get_val_transforms()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, mask = self.dataset[idx]
        size = [self.image_size, self.image_size]
        image = TF.resize(image, size, interpolation=InterpolationMode.BILINEAR)
        mask = TF.resize(mask, size, interpolation=InterpolationMode.NEAREST)
        return self.image_transform(image), mask_to_target(mask, self.num_classes)


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #
def extract_spatial_features(model, x):
    """Return a (B, C, h, w) feature map from the backbone of an ImageSSL model.

    Supports the ResNet18 wrapper and torchvision VisionTransformer from main.py.
    Falls back to the pooled features broadcast as a 1x1 map for unknown backbones.
    """
    backbone = model.backbone

    # ResNet18 wrapper (main.ResNet18) -> torchvision resnet in .backbone
    resnet = getattr(backbone, "backbone", None)
    if resnet is not None and hasattr(resnet, "layer4"):
        x = resnet.relu(resnet.bn1(resnet.conv1(x)))
        x = resnet.maxpool(x)
        x = resnet.layer1(x)
        x = resnet.layer2(x)
        x = resnet.layer3(x)
        return resnet.layer4(x)

    # torchvision VisionTransformer: patch tokens -> grid
    if hasattr(backbone, "_process_input") and hasattr(backbone, "class_token"):
        tokens = backbone._process_input(x)  # (B, N, D)
        cls = backbone.class_token.expand(tokens.shape[0], -1, -1)
        tokens = backbone.encoder(torch.cat([cls, tokens], dim=1))[:, 1:]
        b, n, d = tokens.shape
        side = int(n**0.5)
        return tokens.transpose(1, 2).reshape(b, d, side, side)

    features, _ = model(x)
    return features[:, :, None, None]


# --------------------------------------------------------------------------- #
# Probe
# --------------------------------------------------------------------------- #
class LinearProbe(nn.Module):
    """Per-pixel linear classifier (1x1 conv) on a frozen spatial feature map."""

    def __init__(self, feature_dim, num_classes):
        super().__init__()
        self.bn = nn.BatchNorm2d(feature_dim, affine=False)
        self.classifier = nn.Conv2d(feature_dim, num_classes, kernel_size=1)

    def forward(self, feature_map):
        return self.classifier(self.bn(feature_map))


def probe_logits(model, linear_probe, images, out_size, use_amp=True):
    """Frozen backbone features -> probe logits upsampled to out_size (H, W)."""
    with torch.no_grad(), autocast(images.device.type, enabled=use_amp):
        feature_map = extract_spatial_features(model, images)
    logits = linear_probe(feature_map.float())
    return F.interpolate(logits, size=out_size, mode="bilinear", align_corners=False)


def train_linear_probe(
    model, linear_probe, train_loader, device, epochs=10, lr=1e-3, use_amp=True
):
    """Train the segmentation probe on a frozen backbone.

    train_loader must yield (image tensor, mask tensor), e.g. from
    SegmentationEvalDataset(RockDataset(root, train=True)).
    """
    model.eval()
    optimizer = torch.optim.AdamW(linear_probe.parameters(), lr=lr)
    for _ in tqdm(range(epochs)):
        linear_probe.train()
        for images, masks in train_loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            logits = probe_logits(model, linear_probe, images, masks.shape[-2:], use_amp)
            loss = F.cross_entropy(logits, masks, ignore_index=IGNORE_INDEX)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return linear_probe


@torch.no_grad()
def evaluate_segmentation_probe(
    model, linear_probe, val_loader, device, num_classes, use_amp=True
):
    """Evaluate the probe; returns dict with loss, pixel_acc, miou and per-class iou."""
    model.eval()
    linear_probe.eval()

    total_loss = 0.0
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long, device=device)

    for images, masks in val_loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        logits = probe_logits(model, linear_probe, images, masks.shape[-2:], use_amp)
        total_loss += F.cross_entropy(logits, masks, ignore_index=IGNORE_INDEX).item()

        preds = logits.argmax(1)
        valid = masks != IGNORE_INDEX
        idx = masks[valid] * num_classes + preds[valid]
        confusion += torch.bincount(idx, minlength=num_classes**2).view(
            num_classes, num_classes
        )

    confusion = confusion.double()
    tp = confusion.diag()
    iou = tp / (confusion.sum(0) + confusion.sum(1) - tp).clamp_min(1)
    present = confusion.sum(1) > 0  # only average over classes present in GT

    return {
        "loss": total_loss / max(len(val_loader), 1),
        "pixel_acc": 100.0 * (tp.sum() / confusion.sum().clamp_min(1)).item(),
        "miou": 100.0 * iou[present].mean().item() if present.any() else 0.0,
        "iou_per_class": (100.0 * iou).tolist(),
    }


def evaluate_linear_probe(model, linear_probe, val_loader, device, use_amp=True):
    """main.py-compatible wrapper: returns (pixel accuracy %, avg loss)."""
    num_classes = linear_probe.classifier.out_channels
    metrics = evaluate_segmentation_probe(
        model, linear_probe, val_loader, device, num_classes, use_amp
    )
    return metrics["pixel_acc"], metrics["loss"]

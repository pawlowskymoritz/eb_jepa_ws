"""
Compute per-channel mean and std of an image folder, for transforms.Normalize(mean, std).

Images are found recursively (any folder structure, no labels needed), loaded the
same way as in training (PIL -> RGB, or grayscale with --gray), scaled to [0, 1],
and the statistics are accumulated over all pixels of all images.

Usage (from the repo root):
    uv run python scripts/compute_mean_std.py /path/to/images
    uv run python scripts/compute_mean_std.py /path/to/images --size 96 --workers 8
    uv run python scripts/compute_mean_std.py /path/to/images --max-images 5000
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


class PixelStats(Dataset):
    """Returns per-image (sum, sum of squares, pixel count) per channel."""

    def __init__(self, files, mode, size):
        self.files = files
        self.mode = mode
        self.size = size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert(self.mode)
        if self.size:
            img = img.resize((self.size, self.size), Image.BILINEAR)
        x = np.asarray(img, dtype=np.float64) / 255.0
        if x.ndim == 2:
            x = x[..., None]
        x = x.reshape(-1, x.shape[-1])  # (pixels, channels)
        return (
            torch.from_numpy(x.sum(0)),
            torch.from_numpy((x**2).sum(0)),
            torch.tensor(x.shape[0], dtype=torch.float64),
        )


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("root", type=Path, help="Image folder (searched recursively)")
    p.add_argument("--size", type=int, default=None,
                   help="Resize to SIZE x SIZE before computing (e.g. your training size). Default: original size")
    p.add_argument("--gray", action="store_true", help="Compute single-channel stats (default: RGB)")
    p.add_argument("--max-images", type=int, default=None, help="Use a random subset of this many images")
    p.add_argument("--workers", type=int, default=4, help="DataLoader worker processes")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    files = sorted(f for f in args.root.rglob("*") if f.suffix.lower() in IMG_EXTS)
    if not files:
        raise SystemExit(f"No images with extensions {sorted(IMG_EXTS)} found under {args.root}")
    print(f"Found {len(files)} images under {args.root}")
    if args.max_images and args.max_images < len(files):
        random.Random(args.seed).shuffle(files)
        files = files[: args.max_images]
        print(f"Using random subset of {len(files)} images")

    ds = PixelStats(files, "L" if args.gray else "RGB", args.size)
    # batch_size=None: one image per item, so images of different sizes are fine
    loader = DataLoader(ds, batch_size=None, num_workers=args.workers)

    total, total_sq, n = 0.0, 0.0, 0.0
    for s, sq, cnt in tqdm(loader, desc="Computing"):
        total, total_sq, n = total + s, total_sq + sq, n + cnt

    mean = total / n
    std = (total_sq / n - mean**2).clamp_min(0).sqrt()

    fmt = lambda t: "[" + ", ".join(f"{v:.4f}" for v in t.tolist()) + "]"
    print(f"\nPixels: {int(n):,} per channel")
    print(f"mean: {fmt(mean)}")
    print(f"std:  {fmt(std)}")
    print("\n# Python:")
    print(f"transforms.Normalize({tuple(round(v, 4) for v in mean.tolist())}, "
          f"{tuple(round(v, 4) for v in std.tolist())})")


if __name__ == "__main__":
    main()

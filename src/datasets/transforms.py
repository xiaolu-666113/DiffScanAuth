"""Image transforms preserving gaze coordinate consistency."""

from __future__ import annotations

import torchvision.transforms as T


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_image_transform(
    image_size: int = 384,
    train: bool = True,
    use_aug: bool = True,
):
    """Create deterministic spatial transform + optional photometric augment."""
    tfms = [T.Resize((image_size, image_size), antialias=True)]

    if train and use_aug:
        tfms.extend(
            [
                T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.03),
                T.RandomApply([T.GaussianBlur(kernel_size=3)], p=0.15),
            ]
        )

    tfms.extend([T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    return T.Compose(tfms)

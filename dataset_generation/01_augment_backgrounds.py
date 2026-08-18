"""
Step 1: Pool Background Augmentation
======================================
Takes the original background images and applies:
  1. Rotation (with crop to avoid black corners)
  2. Brightness / contrast jitter
"""

import cv2
import numpy as np
from pathlib import Path
import random
import math

# ----------------------------- Config -----------------------------
RAW_DIR = Path(__file__).parent.parent / "dataset_generation/pool_backgrounds"
OUT_DIR = Path(__file__).parent.parent / "augmented_backgrounds"

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".jfif", ".webp")

TARGET_WIDTH = 1280
TARGET_HEIGHT = 720

N_AUGS_PER_IMAGE = 5
MAX_ROTATION_DEG = 15
BRIGHTNESS_RANGE = (0.65, 1.35)
CONTRAST_RANGE = (0.85, 1.15)
SEED = 42
# --------------------------------------------------------------------

random.seed(SEED)
np.random.seed(SEED)


def standardize_resolution(image: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """
    Standardizes any image to target_w x target_h without stretching.
    A center crop is applied first to match the target aspect ratio,
    then the image is resized.
    """
    h, w = image.shape[:2]
    target_ratio = target_w / target_h
    current_ratio = w / h

    if current_ratio > target_ratio:
        # Image is wider than needed -> crop from the sides
        new_w = int(h * target_ratio)
        x1 = (w - new_w) // 2
        cropped = image[:, x1:x1 + new_w]
    else:
        # Image is taller than needed -> crop from the top and bottom
        new_h = int(w / target_ratio)
        y1 = (h - new_h) // 2
        cropped = image[y1:y1 + new_h, :]

    return cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_AREA)


def rotate_and_crop(image: np.ndarray, angle_deg: float) -> np.ndarray:
    """
    Rotates the image and then crops the largest possible rectangle
    inside the rotated area to avoid black corners.
    """
    h, w = image.shape[:2]
    center = (w / 2, h / 2)

    rot_mat = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated = cv2.warpAffine(
        image,
        rot_mat,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    # Calculate the largest inscribed rectangle
    # that does not contain black corners
    angle_rad = math.radians(abs(angle_deg))

    if w <= 0 or h <= 0:
        return rotated

    # Standard formula for calculating the largest rectangle
    # inside a rotated image while keeping the original aspect ratio
    if w >= h:
        side_long, side_short = w, h
    else:
        side_long, side_short = h, w

    sin_a, cos_a = math.sin(angle_rad), math.cos(angle_rad)

    if side_short <= 2 * sin_a * cos_a * side_long:
        # Large-angle case
        x = 0.5 * side_short
        wr, hr = (
            (x / sin_a, x / cos_a)
            if w >= h
            else (x / cos_a, x / sin_a)
        )
    else:
        cos_2a = cos_a * cos_a - sin_a * sin_a
        wr = (w * cos_a - h * sin_a) / cos_2a
        hr = (h * cos_a - w * sin_a) / cos_2a

    wr, hr = int(wr), int(hr)

    x1 = int(center[0] - wr / 2)
    y1 = int(center[1] - hr / 2)

    x1, y1 = max(0, x1), max(0, y1)

    x2 = min(w, x1 + wr)
    y2 = min(h, y1 + hr)

    cropped = rotated[y1:y2, x1:x2]

    # Resize back to the original image size
    # so the dataset remains consistent
    resized = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

    return resized


def apply_brightness_contrast(
    image: np.ndarray,
    brightness: float,
    contrast: float
) -> np.ndarray:
    """
    brightness: multiplier applied to pixel values (1.0 = unchanged)
    contrast: multiplier around the midpoint (128)
    """
    img = image.astype(np.float32)

    img = (img - 128) * contrast + 128
    img = img * brightness

    return np.clip(img, 0, 255).astype(np.uint8)


def augment_image(image: np.ndarray) -> np.ndarray:
    angle = random.uniform(-MAX_ROTATION_DEG, MAX_ROTATION_DEG)
    brightness = random.uniform(*BRIGHTNESS_RANGE)
    contrast = random.uniform(*CONTRAST_RANGE)

    out = rotate_and_crop(image, angle)
    out = apply_brightness_contrast(out, brightness, contrast)

    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        [
            p for p in RAW_DIR.glob("*")
            if p.suffix.lower() in VALID_EXTENSIONS
        ]
    )

    if not image_paths:
        print(
            f"No images found in {RAW_DIR}. "
            "Make sure you placed the background images there."
        )
        return

    total = 0

    for img_path in image_paths:
        image = cv2.imread(str(img_path))

        if image is None:
            print(f"Failed to open: {img_path.name}")
            continue

        # .jfif is JPEG-based, so save it as .jpg
        # to avoid issues with cv2.imwrite
        save_ext = (
            ".jpg"
            if img_path.suffix.lower() == ".jfif"
            else img_path.suffix
        )

        # Standardize the resolution before applying augmentation
        image = standardize_resolution(
            image,
            TARGET_WIDTH,
            TARGET_HEIGHT
        )

        # Save the standardized original image
        # without rotation or brightness changes
        out_original = OUT_DIR / f"{img_path.stem}_orig{save_ext}"
        cv2.imwrite(str(out_original), image)
        total += 1

        for i in range(N_AUGS_PER_IMAGE):
            aug = augment_image(image)

            out_path = OUT_DIR / f"{img_path.stem}_aug{i:02d}{save_ext}"
            cv2.imwrite(str(out_path), aug)

            total += 1

    print(f"Done. Total images in {OUT_DIR}: {total}")
    print(
        f"({len(image_paths)} original images × "
        f"({N_AUGS_PER_IMAGE} augmentations + 1 original))"
    )


if __name__ == "__main__":
    main()
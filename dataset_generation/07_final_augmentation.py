"""
Final Augmentation Pass
==========================
Takes the final images from yolo_dataset (after Step 6) and applies
additional transformations to make them closer to the quality of 
real underwater camera footage:
  1. Stronger blur (Gaussian + optional light motion blur)
  2. Reduced contrast + lifted black point (milky/washed-out look)
  3. Reduced saturation

The bounding boxes (bboxes) are NOT changed in size, but they are flipped 
horizontally because the image is flipped. Therefore, labels are updated.

Output:
    augmented_final/images/*.jpg
    augmented_final/labels/*.txt   (flipped labels)

Usage:
    python 07_final_augmentation.py
"""

import random
import shutil
from pathlib import Path

import cv2
import numpy as np

# ----------------------------- Config -----------------------------
IN_IMG_DIR = Path(__file__).parent.parent / "dataset_generation/yolo_dataset" / "images"
IN_LABEL_DIR = Path(__file__).parent.parent / "dataset_generation/yolo_dataset" / "labels"

OUT_DIR = Path(__file__).parent.parent / "augmented_final"
OUT_IMG_DIR = OUT_DIR / "images"
OUT_LABEL_DIR = OUT_DIR / "labels"

# 1) Blur
BLUR_SIGMA_RANGE = (1.3, 4.0)          # Much stronger than the original blur in Step 5
MOTION_BLUR_PROBABILITY = 0.30
MOTION_BLUR_KSIZE_RANGE = (7, 17)      # Must be odd, handled in code

# 2) Contrast / washed-out look
CONTRAST_RANGE = (0.55, 0.85)          # Lower than original - clearly reduces contrast
BLACK_POINT_LIFT_RANGE = (0.06, 0.22)  # Shadow lift ratio (higher = foggier/more blurred look)
WASH_COLOR = np.array([190, 190, 185], dtype=np.float32)  # Color added to lift black point (BGR)

# 3) Saturation
SATURATION_RANGE = (0.40, 0.75)        # Reducing color saturation

SEED = 55
# --------------------------------------------------------------------

random.seed(SEED)
np.random.seed(SEED)


def flip_labels_horizontal(label_lines: list) -> list:
    """Flips x_center for each line (class_id xc yc w h) - others remain unchanged"""
    new_lines = []
    for line in label_lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        cls_id = parts[0]
        xc, yc, w, h = map(float, parts[1:5])
        new_xc = 1.0 - xc
        new_lines.append(f"{cls_id} {new_xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    return new_lines


def apply_motion_blur(img: np.ndarray, ksize: int, angle_deg: float) -> np.ndarray:
    kernel = np.zeros((ksize, ksize), dtype=np.float32)
    kernel[ksize // 2, :] = 1.0
    M = cv2.getRotationMatrix2D((ksize / 2, ksize / 2), angle_deg, 1.0)
    kernel = cv2.warpAffine(kernel, M, (ksize, ksize))
    s = kernel.sum()
    if s > 0:
        kernel /= s
    return cv2.filter2D(img, -1, kernel)


def reduce_saturation(img_bgr_uint8: np.ndarray, factor: float) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr_uint8, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] *= factor
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def apply_final_augmentation(img_bgr: np.ndarray) -> np.ndarray:
    img = img_bgr.astype(np.float32)

    # 1) Stronger Blur
    sigma = random.uniform(*BLUR_SIGMA_RANGE)
    ksize = max(3, int(sigma * 4) | 1)
    img = cv2.GaussianBlur(img, (ksize, ksize), sigma)

    if random.random() < MOTION_BLUR_PROBABILITY:
        mk = random.randrange(*MOTION_BLUR_KSIZE_RANGE)
        mk = mk if mk % 2 == 1 else mk + 1
        angle = random.uniform(0, 180)
        img = apply_motion_blur(img, mk, angle)

    # 2) Contrast reduction + black point lift (washed-out / milky)
    contrast = random.uniform(*CONTRAST_RANGE)
    img = (img - 128) * contrast + 128

    lift = random.uniform(*BLACK_POINT_LIFT_RANGE)
    img = img * (1 - lift) + WASH_COLOR * lift

    img_uint8 = np.clip(img, 0, 255).astype(np.uint8)

    # 3) Saturation reduction
    sat_factor = random.uniform(*SATURATION_RANGE)
    img_uint8 = reduce_saturation(img_uint8, sat_factor)

    return img_uint8


def main():
    OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_LABEL_DIR.mkdir(parents=True, exist_ok=True)

    img_paths = sorted(IN_IMG_DIR.glob("*.jpg"))
    if not img_paths:
        print(f"⚠️  No images found in {IN_IMG_DIR} — Run Step 6 first.")
        return

    for img_path in img_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # Horizontal flip first - so the image is not an identical copy of the original
        img = cv2.flip(img, 1)

        aug = apply_final_augmentation(img)
        cv2.imwrite(str(OUT_IMG_DIR / img_path.name), aug, [cv2.IMWRITE_JPEG_QUALITY, 90])

        label_path = IN_LABEL_DIR / f"{img_path.stem}.txt"
        if label_path.exists():
            lines = label_path.read_text(encoding="utf-8").splitlines()
            flipped_lines = flip_labels_horizontal(lines)
            with open(OUT_LABEL_DIR / label_path.name, "w", encoding="utf-8") as f:
                f.write("\n".join(flipped_lines))

    classes_path = IN_IMG_DIR.parent / "classes.txt"
    if classes_path.exists():
        shutil.copy(classes_path, OUT_DIR / "classes.txt")

    print(f"✅ Final augmentation completed for {len(img_paths)} images in {OUT_IMG_DIR}")


if __name__ == "__main__":
    main()
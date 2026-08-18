"""
Step 5: Global Underwater Grading
=================================
Takes each composite (board + background) and applies underwater effects
to the entire image as a single unit:

  1. Color cast / haze (veiling light blend toward a cyan/teal water color)
  2. Light Gaussian blur (simulates underwater scattering)
  3. Sensor/particle noise
  4. Vignette (darkening toward the image edges)
  5. Final brightness/contrast jitter

The board is closer to the camera than the pool background, so the board
receives weaker attenuation and blur than the background.

The bounding boxes do not change here because all transformations are
photometric, not geometric. Therefore, the labels are copied unchanged
from composites/labels to final/labels.

Outputs:
    final/images/composite_XXXX.jpg
    final/labels/composite_XXXX.json
        -> Same labels as composites/labels

Usage:
    python 05_underwater_grading.py
"""

import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

# ----------------------------- Config -----------------------------
IN_IMG_DIR = (
    Path(__file__).parent.parent
    / "composites"
    / "images"
)

IN_LABEL_DIR = (
    Path(__file__).parent.parent
    / "composites"
    / "labels"
)

IN_MASK_DIR = (
    Path(__file__).parent.parent
    / "composites"
    / "masks"
)

OUT_DIR = (
    Path(__file__).parent.parent
    / "final"
)

OUT_IMG_DIR = OUT_DIR / "images"
OUT_LABEL_DIR = OUT_DIR / "labels"


# Water color in BGR format used by OpenCV.
# The range represents a cyan/teal underwater color cast.
WATER_COLOR_B_RANGE = (150, 205)
WATER_COLOR_G_RANGE = (140, 195)
WATER_COLOR_R_RANGE = (55, 110)


# Haze/attenuation strength applied to the background.
# The background is farther from the camera.
HAZE_STRENGTH_RANGE = (0.20, 0.55)


# The board is much closer to the camera than the background,
# so it receives only a fraction of the background attenuation.
#
# For example:
# 0.3 means the board receives 30% of the attenuation
# applied to the background.
BOARD_ATTENUATION_FACTOR_RANGE = (0.20, 0.45)


# Gaussian blur strength for the background.
BLUR_SIGMA_RANGE = (0.4, 2.2)


# The board receives only a fraction of the background blur
# because it is closer to the camera.
BOARD_BLUR_FACTOR_RANGE = (0.25, 0.55)


# Gaussian blur kernel size used to soften the board mask.
# This creates a smooth transition between the board and background.
MASK_FEATHER_KSIZE = 25


# Sensor/particle noise strength.
# Applied uniformly to the entire image.
NOISE_STD_RANGE = (2, 11)


# Vignette strength.
# Applied uniformly to the entire image.
VIGNETTE_STRENGTH_RANGE = (0.0, 0.35)


BRIGHTNESS_RANGE = (0.85, 1.15)
CONTRAST_RANGE = (0.90, 1.10)

SEED = 33

# --------------------------------------------------------------------

random.seed(SEED)
np.random.seed(SEED)


def random_water_color_bgr():
    """
    Generates a random underwater color in BGR format.
    """

    b = random.randint(
        *WATER_COLOR_B_RANGE
    )

    g = random.randint(
        *WATER_COLOR_G_RANGE
    )

    r = random.randint(
        *WATER_COLOR_R_RANGE
    )

    return np.array(
        [b, g, r],
        dtype=np.float32
    )


def apply_haze(
    img: np.ndarray,
    water_color: np.ndarray,
    strength: float
) -> np.ndarray:
    """
    Applies a veiling-light blend toward the underwater color.

    Higher strength produces a stronger underwater color cast.
    """

    return (
        img * (1 - strength)
        + water_color * strength
    )


def apply_vignette(
    img: np.ndarray,
    strength: float
) -> np.ndarray:
    """
    Darkens the image toward its edges.
    """

    if strength <= 0:
        return img

    h, w = img.shape[:2]

    yy, xx = np.mgrid[
        0:h,
        0:w
    ].astype(np.float32)

    cx = w / 2
    cy = h / 2

    dist = np.sqrt(
        (xx - cx) ** 2
        + (yy - cy) ** 2
    )

    max_dist = np.sqrt(
        cx ** 2
        + cy ** 2
    )

    norm_dist = (
        dist
        / max_dist
    )

    mask = (
        1
        - strength
        * (norm_dist ** 2)
    )

    mask = np.clip(
        mask,
        1 - strength,
        1.0
    )

    return (
        img
        * mask[..., None]
    )


def apply_haze_and_blur(
    img_bgr: np.ndarray,
    water_color: np.ndarray,
    haze_strength: float,
    blur_sigma: float
) -> np.ndarray:
    """
    Applies underwater haze followed by Gaussian blur.
    """

    img = apply_haze(
        img_bgr,
        water_color,
        haze_strength
    )

    if blur_sigma > 0.15:

        # Kernel size must be odd.
        ksize = max(
            3,
            int(blur_sigma * 4) | 1
        )

        img = cv2.GaussianBlur(
            img,
            (ksize, ksize),
            blur_sigma
        )

    return img


def apply_underwater_grading(
    img_bgr: np.ndarray,
    board_mask: np.ndarray = None
) -> np.ndarray:
    """
    Applies underwater photometric effects.

    The background receives stronger haze and blur because
    it is farther from the camera.

    The board receives weaker haze and blur because it is
    closer to the camera.
    """

    img = img_bgr.astype(
        np.float32
    )

    h, w = img.shape[:2]

    # --------------------------------------------------
    # Generate random underwater parameters
    # --------------------------------------------------

    water_color = random_water_color_bgr()

    bg_strength = random.uniform(
        *HAZE_STRENGTH_RANGE
    )

    bg_blur_sigma = (
        random.uniform(
            *BLUR_SIGMA_RANGE
        )
        * (0.6 + bg_strength)
    )

    # --------------------------------------------------
    # Apply different underwater effects to the
    # background and board
    # --------------------------------------------------

    if (
        board_mask is not None
        and board_mask.max() > 0
    ):

        # The board is closer to the camera,
        # so it receives weaker attenuation.
        board_factor = random.uniform(
            *BOARD_ATTENUATION_FACTOR_RANGE
        )

        board_strength = (
            bg_strength
            * board_factor
        )

        # The board also receives less blur.
        board_blur_factor = random.uniform(
            *BOARD_BLUR_FACTOR_RANGE
        )

        board_blur_sigma = (
            bg_blur_sigma
            * board_blur_factor
        )

        # Background/far region
        img_far = apply_haze_and_blur(
            img,
            water_color,
            bg_strength,
            bg_blur_sigma
        )

        # Board/near region
        img_near = apply_haze_and_blur(
            img,
            water_color,
            board_strength,
            board_blur_sigma
        )

        # --------------------------------------------------
        # Feather the board mask
        # --------------------------------------------------

        mask_f = (
            board_mask.astype(
                np.float32
            )
            / 255.0
        )

        # Ensure an odd kernel size.
        k = MASK_FEATHER_KSIZE | 1

        mask_f = cv2.GaussianBlur(
            mask_f,
            (k, k),
            k / 4
        )

        mask_3 = mask_f[..., None]

        # Smoothly blend the board and background regions.
        img = (
            img_near * mask_3
            + img_far * (1 - mask_3)
        )

    else:

        # If no board mask exists,
        # treat the entire image as background.
        img = apply_haze_and_blur(
            img,
            water_color,
            bg_strength,
            bg_blur_sigma
        )

    # --------------------------------------------------
    # Camera-level effects
    # --------------------------------------------------

    # Vignette is applied uniformly to the whole image.
    vignette_strength = random.uniform(
        *VIGNETTE_STRENGTH_RANGE
    )

    img = apply_vignette(
        img,
        vignette_strength
    )

    # Sensor/particle noise is applied uniformly.
    noise_std = random.uniform(
        *NOISE_STD_RANGE
    )

    img = (
        img
        + np.random.normal(
            0,
            noise_std,
            img.shape
        )
    )

    # Final brightness/contrast jitter.
    brightness = random.uniform(
        *BRIGHTNESS_RANGE
    )

    contrast = random.uniform(
        *CONTRAST_RANGE
    )

    img = (
        (img - 128)
        * contrast
        + 128
    )

    img = (
        img
        * brightness
    )

    return np.clip(
        img,
        0,
        255
    ).astype(np.uint8)


def main():

    OUT_IMG_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    OUT_LABEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    img_paths = sorted(
        IN_IMG_DIR.glob("*.jpg")
    )

    if not img_paths:

        print(
            f"No composites found in {IN_IMG_DIR}. "
            "Run Step 4 first."
        )

        return

    for img_path in img_paths:

        img = cv2.imread(
            str(img_path)
        )

        if img is None:
            continue

        # --------------------------------------------------
        # Load board mask
        # --------------------------------------------------

        mask_path = (
            IN_MASK_DIR
            / f"{img_path.stem}.png"
        )

        if mask_path.exists():

            board_mask = cv2.imread(
                str(mask_path),
                cv2.IMREAD_GRAYSCALE
            )

        else:

            board_mask = None

        # --------------------------------------------------
        # Apply underwater grading
        # --------------------------------------------------

        graded = apply_underwater_grading(
            img,
            board_mask
        )

        # --------------------------------------------------
        # Save graded image
        # --------------------------------------------------

        cv2.imwrite(
            str(
                OUT_IMG_DIR
                / img_path.name
            ),
            graded,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                95
            ]
        )

        # --------------------------------------------------
        # Copy labels unchanged
        # --------------------------------------------------

        label_path = (
            IN_LABEL_DIR
            / f"{img_path.stem}.json"
        )

        if label_path.exists():

            shutil.copy(
                label_path,
                OUT_LABEL_DIR
                / label_path.name
            )

    print(
        f"Underwater grading completed for "
        f"{len(img_paths)} images in {OUT_IMG_DIR}"
    )


if __name__ == "__main__":
    main()
"""
Step 4: Composite Warped Board onto Pool Background
====================================================
Takes each board from Step 3 (RGBA with alpha) and a random background
from Step 1. The board is resized to a random scale to simulate different
camera distances, placed at a random position on the background, and then
the bounding boxes of all shapes are updated to full-image coordinates.

The script also generates a binary/grayscale mask showing the visible
location of the board in the final composite image.

Outputs:
    composites/images/composite_XXXX.jpg
    composites/labels/composite_XXXX.json
        -> Bounding boxes in full-image coordinates
    composites/masks/composite_XXXX.png
        -> Board mask (0-255)

Usage:
    python 04_composite.py
"""

import json
import random
from pathlib import Path

import cv2
import numpy as np

# ----------------------------- Config -----------------------------
BG_DIR = Path(__file__).parent.parent / "dataset_generation/augmented_backgrounds"

BOARD_IMG_DIR = (
    Path(__file__).parent.parent
    / "dataset_generation/boards_warped"
    / "images"
)

BOARD_LABEL_DIR = (
    Path(__file__).parent.parent
    / "dataset_generation/boards_warped"
    / "labels"
)

OUT_DIR = (
    Path(__file__).parent.parent
    / "composites"
)

OUT_IMG_DIR = OUT_DIR / "images"
OUT_LABEL_DIR = OUT_DIR / "labels"

# Binary/grayscale mask showing the board location
# in the final composite image.
OUT_MASK_DIR = OUT_DIR / "masks"

N_COMPOSITES = 300

# Probability of allowing partial board cropping.
# The remaining images keep the entire board inside the frame.
CROP_ALLOWED_PROBABILITY = 0.30

# When cropping is allowed, the board can become large
# and partially extend beyond the image boundaries.
CROPPED_SCALE_RANGE = (0.60, 1.35)

MAX_OFFSCREEN_FRACTION = 0.4

# When the entire board must remain inside the frame.
FULL_FRAME_SCALE_RANGE = (0.35, 0.90)

# Filtering for shapes that are too heavily cropped
# to be useful for training.
MIN_VISIBLE_PX = 10
MIN_VISIBLE_FRACTION = 0.2

# Slight vertical bias to simulate the board usually
# being located on the pool floor.
VERTICAL_BIAS = 0.55
# 0.5 = center of the image
# > 0.5 = slightly lower position

SEED = 21

# --------------------------------------------------------------------

random.seed(SEED)
np.random.seed(SEED)


def load_backgrounds():
    exts = (
        ".jpg",
        ".jpeg",
        ".png"
    )

    return sorted(
        [
            p
            for p in BG_DIR.glob("*")
            if p.suffix.lower() in exts
        ]
    )


def load_boards():
    return sorted(
        BOARD_IMG_DIR.glob("*.png")
    )


def alpha_composite(
    bg: np.ndarray,
    fg_rgba: np.ndarray,
    x: int,
    y: int
):
    """
    Places the RGBA foreground over the background at position (x, y)
    using proper alpha blending.

    Also returns a grayscale mask (0-255) showing the visible board area.
    """

    fh, fw = fg_rgba.shape[:2]
    bh, bw = bg.shape[:2]

    # Create an empty mask for the full background
    mask = np.zeros(
        (bh, bw),
        dtype=np.uint8
    )

    # Clip any part extending outside the background
    x1 = max(0, x)
    y1 = max(0, y)

    x2 = min(bw, x + fw)
    y2 = min(bh, y + fh)

    if x1 >= x2 or y1 >= y2:
        return bg, mask

    # Corresponding coordinates inside the foreground image
    fg_x1 = x1 - x
    fg_y1 = y1 - y

    fg_x2 = fg_x1 + (x2 - x1)
    fg_y2 = fg_y1 + (y2 - y1)

    fg_crop = fg_rgba[
        fg_y1:fg_y2,
        fg_x1:fg_x2
    ]

    bg_crop = bg[
        y1:y2,
        x1:x2
    ].astype(np.float32)

    # Extract alpha channel
    alpha = (
        fg_crop[:, :, 3:4]
        .astype(np.float32)
        / 255.0
    )

    # Extract foreground color channels
    fg_rgb = fg_crop[
        :, :, :3
    ].astype(np.float32)

    # Alpha blending
    blended = (
        fg_rgb * alpha
        + bg_crop * (1 - alpha)
    )

    out = bg.copy()

    out[
        y1:y2,
        x1:x2
    ] = blended.astype(np.uint8)

    # Store the board alpha values in the mask
    mask[
        y1:y2,
        x1:x2
    ] = fg_crop[:, :, 3]

    return out, mask


def composite_one(
    bg_path: Path,
    board_path: Path
):
    """
    Creates one composite image by placing a randomly scaled
    warped board onto a randomly selected pool background.

    Returns:
        composite_img
        new_label
        board_mask
    """

    bg = cv2.imread(
        str(bg_path)
    )

    board_rgba = cv2.imread(
        str(board_path),
        cv2.IMREAD_UNCHANGED
    )

    label_path = (
        BOARD_LABEL_DIR
        / f"{board_path.stem}.json"
    )

    with open(
        label_path,
        "r",
        encoding="utf-8"
    ) as f:
        board_label = json.load(f)

    bg_h, bg_w = bg.shape[:2]

    board_h, board_w = board_rgba.shape[:2]

    # Decide whether partial cropping is allowed
    allow_crop = (
        random.random()
        < CROP_ALLOWED_PROBABILITY
    )

    if allow_crop:
        scale_range = CROPPED_SCALE_RANGE
        offscreen_fraction = MAX_OFFSCREEN_FRACTION
    else:
        scale_range = FULL_FRAME_SCALE_RANGE
        offscreen_fraction = 0.0

    # Calculate the scale so that the board width
    # becomes a random fraction of the background width
    target_w_ratio = random.uniform(
        *scale_range
    )

    target_w = int(
        bg_w * target_w_ratio
    )

    scale = target_w / board_w

    target_h = max(
        1,
        int(board_h * scale)
    )

    # Use area interpolation when shrinking
    # and linear interpolation when enlarging
    interp = (
        cv2.INTER_AREA
        if scale < 1.0
        else cv2.INTER_LINEAR
    )

    board_resized = cv2.resize(
        board_rgba,
        (target_w, target_h),
        interpolation=interp
    )

    # Maximum amount of the board allowed
    # to extend outside the background
    max_off_x = int(
        target_w
        * offscreen_fraction
    )

    max_off_y = int(
        target_h
        * offscreen_fraction
    )

    x_min = -max_off_x
    x_max = (
        bg_w
        - target_w
        + max_off_x
    )

    if x_max > x_min:
        x = random.randint(
            x_min,
            x_max
        )
    else:
        x = 0

    # Vertical position with a slight bias
    # toward the lower part of the image
    y_min = -max_off_y
    y_max = (
        bg_h
        - target_h
        + max_off_y
    )

    if y_max > y_min:

        center_y = (
            y_min
            + VERTICAL_BIAS
            * (y_max - y_min)
        )

        spread = (
            (y_max - y_min)
            * 0.3
        )

        y = int(
            np.clip(
                np.random.normal(
                    center_y,
                    spread
                ),
                y_min,
                y_max
            )
        )

    else:
        y = 0

    # Composite the board onto the background
    composite_img, board_mask = alpha_composite(
        bg,
        board_resized,
        x,
        y
    )

    new_shapes = []

    for shape in board_label["shapes"]:

        x1, y1, x2, y2 = shape["bbox"]

        # Apply the same scale used for the board,
        # then add the board position offset.
        fx1 = x1 * scale + x
        fy1 = y1 * scale + y

        fx2 = x2 * scale + x
        fy2 = y2 * scale + y

        full_area = (
            max(0.0, fx2 - fx1)
            * max(0.0, fy2 - fy1)
        )

        if full_area <= 0:
            continue

        # Clip the bbox to the final image boundaries
        nx1 = max(0, fx1)
        ny1 = max(0, fy1)

        nx2 = min(bg_w, fx2)
        ny2 = min(bg_h, fy2)

        vis_w = nx2 - nx1
        vis_h = ny2 - ny1

        # Shape is completely outside the image
        if vis_w <= 0 or vis_h <= 0:
            continue

        visible_area = (
            vis_w
            * vis_h
        )

        visible_fraction = (
            visible_area
            / full_area
        )

        # Remove shapes that are too small
        # or too heavily cropped
        if (
            vis_w < MIN_VISIBLE_PX
            or vis_h < MIN_VISIBLE_PX
        ):
            continue

        if visible_fraction < MIN_VISIBLE_FRACTION:
            continue

        new_shape = dict(shape)

        new_shape["bbox"] = [
            round(float(v), 1)
            for v in (
                nx1,
                ny1,
                nx2,
                ny2
            )
        ]

        # Mark whether the shape is partially cropped
        new_shape["truncated"] = (
            visible_fraction < 0.95
        )

        new_shapes.append(
            new_shape
        )

    new_label = {
        "image_width": bg_w,
        "image_height": bg_h,
        "background_source": bg_path.name,
        "board_source": board_path.name,
        "shapes": new_shapes,
    }

    return (
        composite_img,
        new_label,
        board_mask
    )


def main():
    OUT_IMG_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    OUT_LABEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    OUT_MASK_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    backgrounds = load_backgrounds()
    boards = load_boards()

    if not backgrounds:
        print(
            f"No backgrounds found in {BG_DIR}. "
            "Run Step 1 first."
        )
        return

    if not boards:
        print(
            f"No boards found in {BOARD_IMG_DIR}. "
            "Run Step 3 first."
        )
        return

    for i in range(N_COMPOSITES):

        bg_path = random.choice(
            backgrounds
        )

        board_path = random.choice(
            boards
        )

        (
            composite_img,
            new_label,
            board_mask
        ) = composite_one(
            bg_path,
            board_path
        )

        name = f"composite_{i:04d}"

        # Save composite image
        cv2.imwrite(
            str(
                OUT_IMG_DIR
                / f"{name}.jpg"
            ),
            composite_img,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                95
            ]
        )

        # Save board mask
        cv2.imwrite(
            str(
                OUT_MASK_DIR
                / f"{name}.png"
            ),
            board_mask
        )

        # Save labels
        with open(
            OUT_LABEL_DIR
            / f"{name}.json",
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                new_label,
                f,
                ensure_ascii=False,
                indent=2
            )

    print(
        f"Created {N_COMPOSITES} composite images "
        f"in {OUT_IMG_DIR}"
    )


if __name__ == "__main__":
    main()
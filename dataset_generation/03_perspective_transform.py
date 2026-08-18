"""
Step 3: Perspective + Rotation on the Whole Board
===================================================
Takes each board from Step 2 (image + JSON labels) and applies a random
homography transform (rotation + perspective tilt) to simulate the board
floating/tilting in front of the underwater camera.

Important:
The four corners of each shape are tracked through the transform, and a
new axis-aligned bounding box is calculated from the transformed points.
The old bbox is NOT transformed as a single rectangle because that would
produce incorrect results after rotation or perspective transformation.

Outputs:
    boards_warped/images/board_XXXX.png   -> RGBA image
                                              (transparent outside board)
    boards_warped/labels/board_XXXX.json  -> New bounding boxes + canvas size

Usage:
    python 03_perspective_transform.py
"""

import json
import math
import random
from pathlib import Path

import cv2
import numpy as np

# ----------------------------- Config -----------------------------
IN_IMG_DIR = Path(__file__).parent.parent / "dataset_generation/boards" / "images"
IN_LABEL_DIR = Path(__file__).parent.parent / "dataset_generation/boards" / "labels"

OUT_DIR = Path(__file__).parent.parent / "boards_warped"
OUT_IMG_DIR = OUT_DIR / "images"
OUT_LABEL_DIR = OUT_DIR / "labels"

INTENSITY_RANGE = (0.35, 1.0)
# Every board receives a warp with a randomly selected intensity.
# This creates boards with different levels of tilt.

ROTATION_RANGE_DEG = 22
# Maximum in-plane board rotation.
# The actual rotation is multiplied by the random intensity.

PERSPECTIVE_STRENGTH = 0.13
# Maximum perspective tilt/skew as a fraction of the board size.
# The actual strength is multiplied by the random intensity.

CANVAS_PADDING = 20
# Extra pixels around the warped result.

SEED = 11
# --------------------------------------------------------------------

random.seed(SEED)
np.random.seed(SEED)


def random_homography(w: int, h: int, intensity: float = 1.0):
    """
    Returns the source corners of the original board and the destination
    corners after applying random rotation and perspective jitter.

    The transformation does not include any shift or padding yet.

    intensity:
        A multiplier between approximately 0 and 1 that controls the
        transformation strength for each board.
    """
    src = np.array(
        [
            [0, 0],
            [w, 0],
            [w, h],
            [0, h]
        ],
        dtype=np.float32
    )

    # Calculate random rotation angle
    max_angle = ROTATION_RANGE_DEG * intensity

    angle = math.radians(
        random.uniform(-max_angle, max_angle)
    )

    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    rot_mat = np.array(
        [
            [cos_a, -sin_a],
            [sin_a, cos_a]
        ],
        dtype=np.float32
    )

    center = np.array(
        [w / 2, h / 2],
        dtype=np.float32
    )

    # Rotate the four board corners around the center
    rotated = (
        (src - center)
        @ rot_mat.T
        + center
    )

    # Add random perspective jitter
    strength = PERSPECTIVE_STRENGTH * intensity

    jitter = np.random.uniform(
        -strength,
        strength,
        size=(4, 2)
    )

    jitter *= np.array(
        [w, h],
        dtype=np.float32
    )

    dst = rotated + jitter

    return src, dst.astype(np.float32)


def compute_transform_and_canvas(
    src,
    dst,
    padding=CANVAS_PADDING
):
    """
    Computes the homography matrix and shifts the destination points
    so that all transformed coordinates remain positive inside the canvas.

    Returns:
        M        -> Final homography matrix
        canvas_w -> Output canvas width
        canvas_h -> Output canvas height
    """

    xmin = dst[:, 0].min()
    ymin = dst[:, 1].min()

    xmax = dst[:, 0].max()
    ymax = dst[:, 1].max()

    # Shift destination points so the minimum coordinates
    # start at the requested padding
    shift = np.array(
        [
            -xmin + padding,
            -ymin + padding
        ],
        dtype=np.float32
    )

    dst_shifted = dst + shift

    M = cv2.getPerspectiveTransform(
        src,
        dst_shifted
    )

    canvas_w = (
        int(math.ceil(xmax - xmin))
        + 2 * padding
    )

    canvas_h = (
        int(math.ceil(ymax - ymin))
        + 2 * padding
    )

    return M, canvas_w, canvas_h


def transform_points(
    M,
    points: np.ndarray
) -> np.ndarray:
    """
    Transforms a set of 2D points using the homography matrix.

    Input:
        points -> shape (N, 2)

    Output:
        Transformed points -> shape (N, 2)
    """
    pts = points.reshape(
        -1,
        1,
        2
    ).astype(np.float32)

    out = cv2.perspectiveTransform(
        pts,
        M
    )

    return out.reshape(
        -1,
        2
    )


def warp_one_board(
    img_path: Path,
    label_path: Path
):
    board_bgr = cv2.imread(
        str(img_path),
        cv2.IMREAD_UNCHANGED
    )

    if board_bgr is None:
        return None, None

    h, w = board_bgr.shape[:2]

    # Convert BGR image to BGRA by adding a fully opaque alpha channel
    if board_bgr.shape[2] == 3:
        alpha = np.full(
            (h, w, 1),
            255,
            dtype=np.uint8
        )

        board_rgba = np.concatenate(
            [board_bgr, alpha],
            axis=2
        )

    else:
        board_rgba = board_bgr

    # Load the original labels
    with open(
        label_path,
        "r",
        encoding="utf-8"
    ) as f:
        label_data = json.load(f)

    # Select a random warp intensity for this board
    intensity = random.uniform(
        *INTENSITY_RANGE
    )

    src, dst = random_homography(
        w,
        h,
        intensity
    )

    M, canvas_w, canvas_h = compute_transform_and_canvas(
        src,
        dst
    )

    # Apply the perspective transformation
    warped = cv2.warpPerspective(
        board_rgba,
        M,
        (canvas_w, canvas_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )

    new_shapes = []

    for shape in label_data["shapes"]:

        x1, y1, x2, y2 = shape["bbox"]

        # Get the four corners of the original bbox
        corners = np.array(
            [
                [x1, y1],
                [x2, y1],
                [x2, y2],
                [x1, y2]
            ],
            dtype=np.float32
        )

        # Transform all four corners
        transformed = transform_points(
            M,
            corners
        )

        # Calculate a new axis-aligned bounding box
        # from the transformed corners
        nx1 = transformed[:, 0].min()
        ny1 = transformed[:, 1].min()

        nx2 = transformed[:, 0].max()
        ny2 = transformed[:, 1].max()

        # Clip coordinates to the canvas boundaries
        nx1 = max(0, nx1)
        ny1 = max(0, ny1)

        nx2 = min(canvas_w, nx2)
        ny2 = min(canvas_h, ny2)

        # Preserve all original shape metadata
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

        new_shapes.append(new_shape)

    new_label = {
        "canvas_width": canvas_w,
        "canvas_height": canvas_h,
        "warp_intensity": round(
            intensity,
            2
        ),
        "shapes": new_shapes,
    }

    return warped, new_label


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
        IN_IMG_DIR.glob("*.png")
    )

    if not img_paths:
        print(
            f"No boards found in {IN_IMG_DIR}. "
            "Run Step 2 first."
        )
        return

    count = 0

    for img_path in img_paths:

        label_path = (
            IN_LABEL_DIR
            / f"{img_path.stem}.json"
        )

        if not label_path.exists():
            continue

        warped, new_label = warp_one_board(
            img_path,
            label_path
        )

        if warped is None:
            continue

        cv2.imwrite(
            str(
                OUT_IMG_DIR
                / f"{img_path.stem}.png"
            ),
            warped
        )

        with open(
            OUT_LABEL_DIR
            / f"{img_path.stem}.json",
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                new_label,
                f,
                ensure_ascii=False,
                indent=2
            )

        count += 1

    print(
        f"Applied perspective warp to {count} boards "
        f"in {OUT_IMG_DIR}"
    )


if __name__ == "__main__":
    main()
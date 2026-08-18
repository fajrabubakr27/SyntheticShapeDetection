"""
Step 2: Base Board Generation
================================
Generates square images representing the white plastic board underwater,
containing 1-3 shapes from the 5 classes. Each shape gets a random color
(domain randomization) within a class-specific color range.

Outputs:
    boards/images/board_XXXX.png   -> Board image
    boards/labels/board_XXXX.json  -> Bounding box for each shape
                                       (board pixel coordinates)

Usage:
    python 02_generate_boards.py
"""

import json
import math
import random
import colorsys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

# ----------------------------- Config -----------------------------
OUT_DIR = Path(__file__).parent.parent / "boards"
IMG_DIR = OUT_DIR / "images"
LABEL_DIR = OUT_DIR / "labels"

BOARD_SIZE = 800
N_BOARDS = 300

SHAPES_PER_BOARD_RANGE = (1, 3)
SHAPE_SIZE_RANGE = (0.16, 0.42)
SUPERSAMPLE = 4

BOARD_GREY_RANGE = (215, 252)
BOARD_NOISE_STD = 5
LIGHTING_GRADIENT_STRENGTH_RANGE = (0.0, 0.35)
SHAPE_LOCAL_ROTATION_JITTER = 6
FRAME_THICKNESS_RANGE = (15, 35)
FRAME_GREY_RANGE = (35, 85)
INTERIOR_MARGIN_EXTRA = 12
OVERLAP_MARGIN = 12
MAX_PLACEMENT_ATTEMPTS = 60

CLASSES = ["circle", "triangle", "rectangle", "square", "star"]

# hue_center, hue_spread in degrees (0-360) for each class
# These values are based on the original colors
# (red/blue/green/yellow/orange), but the spread is wide
# enough to prevent the model from relying on a single color shade.
HUE_RANGES = {
    "circle":    (0,   18),   # Red
    "triangle":  (215, 25),   # Blue
    "rectangle": (120, 22),   # Green
    "square":    (50,  14),   # Yellow
    "star":      (28,  14),   # Orange
}

SATURATION_RANGE = (0.55, 0.95)
VALUE_RANGE = (0.55, 0.95)

SEED = 7
# --------------------------------------------------------------------

random.seed(SEED)
np.random.seed(SEED)


def random_color_for_class(cls_name: str) -> tuple:
    hue_center, hue_spread = HUE_RANGES[cls_name]

    hue = (
        hue_center
        + random.uniform(-hue_spread, hue_spread)
    ) % 360

    sat = random.uniform(*SATURATION_RANGE)
    val = random.uniform(*VALUE_RANGE)

    r, g, b = colorsys.hsv_to_rgb(
        hue / 360,
        sat,
        val
    )

    return (
        int(r * 255),
        int(g * 255),
        int(b * 255)
    )


def star_points(cx, cy, r_outer, r_inner, n_points=5):
    pts = []

    for i in range(n_points * 2):
        angle = math.pi / n_points * i - math.pi / 2
        r = r_outer if i % 2 == 0 else r_inner

        pts.append(
            (
                cx + r * math.cos(angle),
                cy + r * math.sin(angle)
            )
        )

    return pts


def render_shape(
    cls_name: str,
    size: int,
    color: tuple,
    angle: float
) -> Image.Image:
    """
    Renders the shape on a transparent canvas using supersampling
    for better anti-aliasing, rotates it, then crops it to the
    tightest bounding box around the actual shape pixels.
    The returned image is exactly the size of the shape's bbox.
    """
    ss = SUPERSAMPLE
    canvas = size * ss

    img = Image.new(
        "RGBA",
        (canvas, canvas),
        (0, 0, 0, 0)
    )

    draw = ImageDraw.Draw(img)

    cx = cy = canvas / 2
    r = canvas * 0.42
    fill = color + (255,)

    if cls_name == "circle":
        draw.ellipse(
            [cx - r, cy - r, cx + r, cy + r],
            fill=fill
        )

    elif cls_name == "square":
        half = r * 0.82

        draw.rectangle(
            [cx - half, cy - half, cx + half, cy + half],
            fill=fill
        )

    elif cls_name == "rectangle":
        w_half = r * 0.95
        h_half = r * 0.55

        draw.rectangle(
            [cx - w_half, cy - h_half, cx + w_half, cy + h_half],
            fill=fill
        )

    elif cls_name == "triangle":
        pts = [
            (cx, cy - r),
            (cx - r * 0.87, cy + r * 0.5),
            (cx + r * 0.87, cy + r * 0.5)
        ]

        draw.polygon(pts, fill=fill)

    elif cls_name == "star":
        pts = star_points(
            cx,
            cy,
            r,
            r * 0.45,
            5
        )

        draw.polygon(pts, fill=fill)

    else:
        raise ValueError(f"Unknown class: {cls_name}")

    img = img.rotate(
        angle,
        resample=Image.BICUBIC,
        expand=True
    )

    bbox = img.getbbox()

    if bbox is None:
        return img

    img = img.crop(bbox)

    final_w = max(1, img.width // ss)
    final_h = max(1, img.height // ss)

    img = img.resize(
        (final_w, final_h),
        Image.LANCZOS
    )

    return img


def boxes_overlap(a, b, margin=OVERLAP_MARGIN) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    return not (
        ax2 + margin < bx1
        or bx2 + margin < ax1
        or ay2 + margin < by1
        or by2 + margin < ay1
    )


def make_board_background() -> Image.Image:
    grey = random.randint(*BOARD_GREY_RANGE)

    arr = np.full(
        (BOARD_SIZE, BOARD_SIZE, 3),
        grey,
        dtype=np.float32
    )

    # Random lighting gradient to simulate
    # uneven underwater lighting across the board
    strength = random.uniform(
        *LIGHTING_GRADIENT_STRENGTH_RANGE
    )

    if strength > 0:
        angle = random.uniform(
            0,
            2 * math.pi
        )

        yy, xx = np.mgrid[
            0:BOARD_SIZE,
            0:BOARD_SIZE
        ]

        proj = (
            xx * math.cos(angle)
            + yy * math.sin(angle)
        ) / (
            BOARD_SIZE * math.sqrt(2)
        )

        proj = proj - proj.mean()

        proj = proj / (
            np.abs(proj).max() + 1e-6
        )

        # Multiplier around 1.0
        gradient = 1.0 + strength * proj

        arr = arr * gradient[..., None]

    # Add subtle noise to simulate plastic texture
    noise = np.random.normal(
        0,
        BOARD_NOISE_STD,
        arr.shape
    )

    arr = np.clip(
        arr + noise,
        0,
        255
    ).astype(np.uint8)

    return Image.fromarray(arr)


def draw_frame(board: Image.Image) -> int:
    draw = ImageDraw.Draw(board)

    thickness = random.randint(
        *FRAME_THICKNESS_RANGE
    )

    grey = random.randint(
        *FRAME_GREY_RANGE
    )

    color = (grey, grey, grey)

    draw.rectangle(
        [0, 0, BOARD_SIZE - 1, BOARD_SIZE - 1],
        outline=color,
        width=thickness,
    )

    return thickness


def generate_one_board(idx: int):
    board = make_board_background()

    frame_thickness = draw_frame(board)

    margin = (
        frame_thickness
        + INTERIOR_MARGIN_EXTRA
    )

    n_shapes = random.randint(
        *SHAPES_PER_BOARD_RANGE
    )

    chosen_classes = random.choices(
        CLASSES,
        k=n_shapes
    )

    placed_boxes = []
    shapes_data = []

    for cls_name in chosen_classes:

        for _ in range(MAX_PLACEMENT_ATTEMPTS):

            size = random.randint(
                int(
                    BOARD_SIZE
                    * SHAPE_SIZE_RANGE[0]
                ),
                int(
                    BOARD_SIZE
                    * SHAPE_SIZE_RANGE[1]
                ),
            )

            angle = random.uniform(
                -SHAPE_LOCAL_ROTATION_JITTER,
                SHAPE_LOCAL_ROTATION_JITTER
            )

            color = random_color_for_class(
                cls_name
            )

            shape_img = render_shape(
                cls_name,
                size,
                color,
                angle
            )

            sw, sh = shape_img.size

            max_x = BOARD_SIZE - margin - sw
            max_y = BOARD_SIZE - margin - sh

            if max_x <= margin or max_y <= margin:
                continue

            x = random.randint(
                margin,
                max_x
            )

            y = random.randint(
                margin,
                max_y
            )

            box = (
                x,
                y,
                x + sw,
                y + sh
            )

            if any(
                boxes_overlap(box, b)
                for b in placed_boxes
            ):
                continue

            board.paste(
                shape_img,
                (x, y),
                shape_img
            )

            placed_boxes.append(box)

            shapes_data.append({
                "class_name": cls_name,
                "class_id": CLASSES.index(cls_name),
                "bbox": [
                    x,
                    y,
                    x + sw,
                    y + sh
                ],
                "color_rgb": list(color),
                "rotation_deg": round(angle, 1),
            })

            break

    return board, shapes_data


def main():
    IMG_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    LABEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    for i in range(N_BOARDS):

        board, shapes_data = generate_one_board(i)

        name = f"board_{i:04d}"

        board.save(
            IMG_DIR / f"{name}.png"
        )

        with open(
            LABEL_DIR / f"{name}.json",
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                {
                    "board_size": BOARD_SIZE,
                    "shapes": shapes_data,
                },
                f,
                ensure_ascii=False,
                indent=2
            )

    print(
        f"Generated {N_BOARDS} boards in {IMG_DIR}"
    )

    print(
        f"Labels (bboxes) saved in {LABEL_DIR}"
    )


if __name__ == "__main__":
    main()
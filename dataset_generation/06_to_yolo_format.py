"""
Step 6: Convert to YOLO Format (No Split)
==========================================
Takes the final images from Step 5 (final/images) and their labels
(final/labels, using x1,y1,x2,y2 pixel coordinates), and converts
each bounding box to YOLO format:

    class_id x_center y_center width height

All coordinate values are normalized to the range 0-1.

No train/val/test split is performed here. The split will be handled
by a separate script (07_split_dataset.py) after all additional
augmentation is completed and the bounding boxes have been
visually verified.

Outputs:
    yolo_dataset/images/*.jpg
        -> Copies of final/images

    yolo_dataset/labels/*.txt
        -> YOLO-format labels

    yolo_dataset/classes.txt
        -> Class names in the correct order

Usage:
    python 06_to_yolo.py
"""

import json
import shutil
from pathlib import Path

# ----------------------------- Config -----------------------------
IN_IMG_DIR = (
    Path(__file__).parent.parent
    / "dataset_generation/final"
    / "images"
)

IN_LABEL_DIR = (
    Path(__file__).parent.parent
    / "dataset_generation/final"
    / "labels"
)

OUT_DIR = (
    Path(__file__).parent.parent
    / "yolo_dataset"
)

OUT_IMG_DIR = OUT_DIR / "images"
OUT_LABEL_DIR = OUT_DIR / "labels"

# The order must exactly match the class IDs
# used in Step 2.
CLASSES = [
    "circle",
    "triangle",
    "rectangle",
    "square",
    "star"
]

# --------------------------------------------------------------------


def bbox_to_yolo(
    bbox,
    img_w,
    img_h
):
    """
    Converts a bounding box from:

        x1, y1, x2, y2

    in pixel coordinates to:

        x_center, y_center, width, height

    normalized to the range 0-1.
    """

    x1, y1, x2, y2 = bbox

    # Calculate normalized center coordinates
    x_center = (
        (x1 + x2) / 2
        / img_w
    )

    y_center = (
        (y1 + y2) / 2
        / img_h
    )

    # Calculate normalized width and height
    width = (
        (x2 - x1)
        / img_w
    )

    height = (
        (y2 - y1)
        / img_h
    )

    # Keep all values within [0, 1]
    x_center = min(
        max(x_center, 0.0),
        1.0
    )

    y_center = min(
        max(y_center, 0.0),
        1.0
    )

    width = min(
        max(width, 0.0),
        1.0
    )

    height = min(
        max(height, 0.0),
        1.0
    )

    return (
        x_center,
        y_center,
        width,
        height
    )


def convert_one_label(
    label_path: Path
) -> list:
    """
    Converts one JSON label file into YOLO-format lines.
    """

    with open(
        label_path,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    img_w = data["image_width"]
    img_h = data["image_height"]

    lines = []

    for shape in data["shapes"]:

        class_id = shape["class_id"]

        xc, yc, w, h = bbox_to_yolo(
            shape["bbox"],
            img_w,
            img_h
        )

        lines.append(
            f"{class_id} "
            f"{xc:.6f} "
            f"{yc:.6f} "
            f"{w:.6f} "
            f"{h:.6f}"
        )

    return lines


def main():

    # Create output directories
    OUT_IMG_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    OUT_LABEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Find all final images
    img_paths = sorted(
        IN_IMG_DIR.glob("*.jpg")
    )

    if not img_paths:

        print(
            f"No images found in {IN_IMG_DIR}. "
            "Run Step 5 first."
        )

        return

    total_written = 0
    total_empty = 0

    for img_path in img_paths:

        # Find the corresponding JSON label
        label_path = (
            IN_LABEL_DIR
            / f"{img_path.stem}.json"
        )

        if not label_path.exists():
            continue

        # Copy image to the YOLO dataset
        shutil.copy(
            img_path,
            OUT_IMG_DIR
            / img_path.name
        )

        # Convert JSON annotations to YOLO format
        lines = convert_one_label(
            label_path
        )

        # Images without objects are valid negative examples
        if not lines:
            total_empty += 1

        # Save YOLO label file
        with open(
            OUT_LABEL_DIR
            / f"{img_path.stem}.txt",
            "w",
            encoding="utf-8"
        ) as f:

            f.write(
                "\n".join(lines)
            )

        total_written += 1

    # Save class names
    with open(
        OUT_DIR / "classes.txt",
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "\n".join(CLASSES)
        )

    print(
        f"Converted {total_written} images "
        f"to YOLO format in {OUT_DIR}"
    )

    if total_empty:

        print(
            f"Note: {total_empty} images contain "
            "no visible objects (negative examples)."
        )

    print(
        f"Classes in order: {CLASSES}"
    )

    print(
        "No train/val/test split was performed. "
        "The split will be created after additional "
        "augmentation and visualization."
    )


if __name__ == "__main__":
    main()
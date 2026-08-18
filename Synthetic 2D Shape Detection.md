# Synthetic 2D Shape Detection

This project builds a synthetic object-detection dataset for five geometric shapes, trains a YOLO26s detector on synthetic images, and evaluates whether the learned detector generalizes to real-world/on-site images that were not used during training.

> **Main question:** Are the synthetic images realistic enough to represent the visual conditions encountered in the real underwater/on-site environment?

The target classes are **circle**, **triangle**, **rectangle**, **square**, and **star**. The synthetic generation pipeline was implemented primarily with **Python, OpenCV, NumPy, Pillow, and custom image-processing utilities**. The training stage used **Ultralytics YOLO26s** on Google Colab with a Tesla T4 GPU. YOLO-compatible bounding-box labels were generated in the standard normalized format described in the Ultralytics object-detection dataset documentation [1].

## Project Objectives

The project follows three requirements:

| Requirement | Implementation status |
|---|---|
| Dataset creation | Completed through a multi-stage synthetic-data pipeline. |
| Model training without online training tools | Completed on Google Colab using a T4 GPU. |
| Evaluation on an OnSite dataset that was excluded from training | Completed using 22 manually annotated real images exported from CVAT in YOLO format. |

The OnSite images were kept separate from the synthetic training and validation data. They were annotated after the synthetic dataset was prepared and used only for external evaluation.

## Dataset and Classes

The synthetic dataset contains **600 images** after merging the original and augmented image sets. It was split into **480 training images** and **120 validation images**. The five classes are encoded as follows:

| Class ID | Class name |
|---:|---|
| 0 | circle |
| 1 | triangle |
| 2 | rectangle |
| 3 | square |
| 4 | star |

The training and validation labels were checked programmatically. The supplied notebook reports matching image/label counts, no missing image-label pairs, and no invalid YOLO label lines.

| Split | Images | Labels | Reported instances |
|---|---:|---:|---:|
| Train | 480 | 480 | 839 |
| Validation | 120 | 120 | 213 |
| **Total synthetic data** | **600** | **600** | **1,052** |

The OnSite evaluation set contains **22 real images and 24 annotated shape instances**. Since this set is small, its metrics should be interpreted as an initial domain-generalization assessment rather than a statistically complete estimate of production performance.

## Synthetic Data-Generation Pipeline

The dataset-generation process is deliberately staged so that geometry, viewpoint, compositing, underwater appearance, annotation conversion, and final augmentation can be inspected independently.

### 1. Background augmentation — `01_augment_backgrounds.py`

The original pool/background images are standardized to **1280 × 720** without stretching. A center crop is applied when the source aspect ratio does not match the target ratio. Each background receives an unchanged standardized copy and five augmented variants.

The augmentation includes random in-plane rotation up to ±15 degrees, brightness variation in the range 0.65–1.35, and contrast variation in the range 0.85–1.15. Rotated images are cropped to reduce invalid corners and are then resized back to the target resolution. This stage creates background diversity before any shape is inserted.

### 2. Base board and shape generation — `02_generate_boards.py`

The script generates **300 board images**, each representing a white plastic board containing one to three shapes. The board is rendered at 800 × 800 pixels with a random gray level, subtle pixel noise, a random lighting gradient, and a dark frame with variable thickness.

Each shape is rendered at high resolution using supersampling and then downsampled for smoother edges. The shape position, size, color, and local rotation are randomized. Shapes are selected from the five target classes and are placed with an overlap constraint so that multiple objects can appear in one image without excessive occlusion.

Class-specific color ranges are used for domain randomization. For example, circles are sampled around red hues, triangles around blue hues, rectangles around green hues, squares around yellow hues, and stars around orange hues. This helps the model see more than one exact RGB value per class, although it also creates a potential shortcut: the class may remain correlated with color.

The script stores the original pixel-coordinate bounding boxes and metadata such as class ID, color, and rotation angle in JSON files.

### 3. Perspective and rotation — `03_perspective_transform.py`

Each board is transformed using a random homography. The transformation combines an in-plane rotation of up to approximately ±22 degrees with random perspective jitter whose maximum strength is 13% of the board dimensions.

A key implementation detail is that the four corners of every original shape bounding box are transformed individually. A new axis-aligned bounding box is then computed from the transformed points. This is more accurate than transforming the old rectangle as if it were still axis-aligned after rotation and perspective distortion.

### 4. Board-to-background compositing — `04_composite.py`

The warped board is alpha-composited onto a randomly selected augmented pool background. The board scale is randomized to simulate different camera distances. In 30% of cases, partial board cropping is allowed, with a maximum off-screen fraction of 40%. In the remaining cases, the board is kept inside the image frame.

The board position is randomized with a slight vertical bias toward the lower portion of the image. Shape bounding boxes are scaled, translated, clipped to the image boundaries, and filtered when the visible region becomes too small. A board mask is also saved for the next stage so that near-board and far-background effects can be applied separately.

### 5. Underwater grading — `05_underwater_grading.py`

This stage attempts to approximate underwater image formation using photometric effects rather than geometry changes. Because no bounding-box coordinates change, the labels are copied unchanged from the compositing stage.

The implemented effects are:

| Effect | Purpose |
|---|---|
| Cyan/teal veiling-light blend | Simulates underwater color cast and haze. |
| Region-dependent blur | Applies stronger degradation to the background and weaker degradation to the board. |
| Board-mask feathering | Creates a smooth transition between board and background. |
| Sensor/particle noise | Adds image-level noise. |
| Vignette | Darkens image edges. |
| Brightness and contrast jitter | Simulates camera exposure variation. |

The current configuration uses background haze strength of 0.20–0.55, background blur sigma of 0.4–2.2, board attenuation factor of 0.20–0.45, board blur factor of 0.25–0.55, noise standard deviation of 2–11, and vignette strength of 0–0.35.

### 6. YOLO conversion — `06_to_yolo_format.py`

The JSON annotations are converted from pixel-coordinate boxes `(x1, y1, x2, y2)` to normalized YOLO lines:

```text
class_id x_center y_center width height
```

The conversion clips the normalized values to the interval [0, 1] and writes the class order to `classes.txt`.

### 7. Final augmentation — `07_final_augmentation.py`

A second augmentation pass is applied to increase the range of degraded underwater appearances. The image is horizontally flipped, and the x-coordinate of every bounding box is updated using `x_center_new = 1 - x_center`.

The visual transformations include stronger Gaussian blur, optional motion blur, contrast reduction, black-point lifting toward a pale wash color, and saturation reduction. This stage is intended to create milky, low-contrast, and motion-affected examples that are difficult for a detector but plausible in underwater footage.

### 8. Dataset merging — `08_merge_datasets.py`

The original YOLO dataset and the final augmented dataset are merged into one directory. Prefixes such as `org_` and `agm_` are added to prevent filename collisions and to preserve the origin of each image.

### 9. Annotation visualization — `09_visualization.py`

Random YOLO samples are rendered with bounding boxes and class names. This step is important because it verifies both the label conversion and the geometric updates after flipping, rotation, perspective transformation, scaling, and cropping.

### 10. Train/validation split — `10_split_dataset.py`

The merged dataset is randomly divided into 80% training and 20% validation subsets. The OnSite dataset is intentionally not included in this split. It remains an external evaluation set.

## Model Training

The detector was trained in Google Colab using Ultralytics **YOLO26s** with pretrained weights. The training configuration was:

| Parameter | Value |
|---|---:|
| Model | `yolo26s.pt` |
| Epochs | 50 |
| Image size | 640 × 640 |
| Batch size | 16 |
| Device | Tesla T4 GPU |
| Workers | 2 |
| Early-stopping patience | 15 |
| Pretrained weights | Enabled |
| Validation during training | Enabled |
| Output checkpoint | `best.pt` |

The notebook reports Ultralytics 8.4.121, PyTorch 2.11.0 with CUDA support, and a YOLO26s model with approximately 9.95 million trainable parameters before fusion. Ultralytics also enabled its built-in low-probability Albumentations operations during training, including blur, median blur, grayscale conversion, and CLAHE. Albumentations is designed to keep bounding boxes synchronized with spatial image transformations when configured with the corresponding box format [2].

## Synthetic Validation Results

The best checkpoint achieved the following results on the held-out synthetic validation split:

| Class | Precision | Recall | mAP@0.50 | mAP@0.50:0.95 |
|---|---:|---:|---:|---:|
| Circle | 0.977 | 0.923 | 0.985 | 0.933 |
| Triangle | 0.986 | 1.000 | 0.995 | 0.911 |
| Rectangle | 1.000 | 0.968 | 0.994 | 0.951 |
| Square | 1.000 | 0.995 | 0.995 | 0.974 |
| Star | 0.963 | 1.000 | 0.994 | 0.919 |
| **Overall** | **0.985** | **0.977** | **0.993** | **0.938** |

The high synthetic validation scores show that the model learned the synthetic visual distribution very effectively. They do **not**, by themselves, prove that the synthetic distribution is realistic. A model can perform extremely well on a validation set when the validation images share the same rendering rules, color conventions, backgrounds, and degradation assumptions as the training images.

## OnSite Evaluation

The OnSite set contains 22 real images and 24 annotated instances. It was used only after training and was exported from CVAT in YOLO format. The evaluation screenshot reports:

| Class | Images | Instances | Precision | Recall | mAP@0.50 | mAP@0.50:0.95 |
|---|---:|---:|---:|---:|---:|---:|
| Circle | 7 | 7 | 0.828 | 0.857 | 0.912 | 0.591 |
| Triangle | 5 | 5 | 0.746 | 0.800 | 0.898 | 0.624 |
| Rectangle | 5 | 5 | 1.000 | 0.729 | 0.795 | 0.698 |
| Square | 2 | 2 | 1.000 | 0.943 | 0.995 | 0.846 |
| Star | 5 | 5 | 0.883 | 1.000 | 0.995 | 0.369 |
| **Overall** | **22** | **24** | **0.891** | **0.866** | **0.912** | **0.626** |

The large gap between synthetic validation and OnSite performance is the most important result of the project. The synthetic split reaches **0.993 mAP@0.50**, whereas the OnSite evaluation reaches **0.912 mAP@0.50** and **0.626 mAP@0.50:0.95**. This indicates that the detector transfers reasonably well at a coarse localization threshold, but its localization quality and confidence are substantially less stable in the real domain.

## Generalization Analysis: Is the Synthetic Data Realistic Enough?

The synthetic data is **partially realistic**. It successfully teaches the model the basic geometry of the five classes, as shown by the OnSite results: **0.891 precision, 0.866 recall, and 0.912 mAP@0.50**.

However, the hardest underwater images remain challenging. Cyan/green haze, low contrast, desaturation, and small objects often lead to low-confidence or missed detections, such as the small circle in `clean_35247.jpg`. This indicates that the main issue is not only classification, but also object visibility and separation from the background.

The performance gap is mainly caused by three factors: real underwater degradation is more complex than the current global haze, blur, noise, and color effects; the board is protected from much of the degradation, leaving the shapes clearer than in real images; and class-specific colors may encourage the model to rely on color instead of shape geometry. In addition, the OnSite set contains only 22 images, so per-class results should be interpreted cautiously.

Overall, the synthetic dataset is useful for synthetic-to-real transfer, but it does not yet represent the most difficult underwater conditions realistically enough.

## Suggested Repository Structure

```text
project_root/
├── dataset_generation/
│   ├── 01_augment_backgrounds.py
│   ├── 02_generate_boards.py
│   ├── 03_perspective_transform.py
│   ├── 04_composite.py
│   ├── 05_underwater_grading.py
│   ├── 06_to_yolo_format.py
│   ├── 07_final_augmentation.py
│   ├── 08_merge_datasets.py
│   ├── 09_visualization.py
│   └── 10_split_dataset.py
├── dataset/
│   ├── data.yaml
│   ├── train/images/
│   ├── train/labels/
│   ├── val/images/
│   └── val/labels/
├── OnSite/
│   ├── data.yaml
│   ├── images/
│   └── labels/
├── training/
│   └── train.ipynb
├── model/
│   └── best.pt
└── README.md
```

## Definition of Done

- [x] Synthetic dataset creation completed.
- [x] Five target classes implemented: circle, triangle, rectangle, square, and star.
- [x] Variations added for backgrounds, lighting, scale, rotation, perspective, cropping, blur, noise, contrast, saturation, and multiple objects.
- [x] YOLO-format annotations generated and visually checked.
- [x] Model trained on Google Colab without using the OnSite images for training.
- [x] OnSite images annotated using CVAT and exported in YOLO format.
- [x] Evaluation performed on the held-out OnSite dataset.
- [x] Synthetic-versus-real generalization analyzed.
- [ ] Optional next step: repeat the evaluation after improving underwater grading and compare the results through an ablation table.

## Conclusion

This project demonstrates a complete synthetic-to-real object-detection workflow. The model learns the five geometric classes very well within the synthetic domain and achieves meaningful transfer to real OnSite images. Nevertheless, the performance gap between the synthetic validation set and the underwater OnSite images shows that the current rendering pipeline underestimates the degradation affecting the object-bearing board and its shapes.

The most defensible conclusion is therefore: **the synthetic dataset is useful and partially realistic, but it is not yet sufficiently realistic for the most degraded underwater conditions**. Increasing board-level attenuation, expanding the diversity of haze and color distortion, reducing class-color shortcuts, and evaluating on a larger real set are the highest-value next steps.

## References

[1]: https://docs.ultralytics.com/datasets/detect "Ultralytics: Object Detection Datasets Overview"
[2]: https://albumentations.ai/docs/3-basic-usage/bounding-boxes-augmentations/ "Albumentations: Bounding Box Augmentations"
[3]: https://opencv-opencv.mintlify.app/ "OpenCV Documentation"
[4]: https://docs.ultralytics.com/modes/train "Ultralytics: Model Training"
[5]: https://docs.ultralytics.com/modes/val "Ultralytics: Model Validation"
[6]: https://docs.cvat.ai/docs/dataset_management/formats/format-yolo/ "CVAT: YOLO Dataset Format"

---


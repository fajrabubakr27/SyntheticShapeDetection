import cv2
import matplotlib.pyplot as plt
import os
import random

def visualize_yolo_data(data_dir, num_samples=4):
    images_dir = os.path.join(data_dir, 'images')
    labels_dir = os.path.join(data_dir, 'labels')
    
    classes = []
    if os.path.exists(os.path.join(data_dir, 'classes.txt')):
        with open(os.path.join(data_dir, 'classes.txt'), 'r') as f:
            classes = [line.strip() for line in f.readlines()]

    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    samples = random.sample(image_files, min(num_samples, len(image_files)))

    plt.figure(figsize=(16, 10))

    for i, img_name in enumerate(samples):
        img_path = os.path.join(images_dir, img_name)
        label_path = os.path.join(labels_dir, os.path.splitext(img_name)[0] + '.txt')

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, _ = image.shape

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = line.split()
                    class_id = int(parts[0])
                    x_center, y_center, w, h = map(float, parts[1:])
                    
                    x1 = int((x_center - w/2) * width)
                    y1 = int((y_center - h/2) * height)
                    x2 = int((x_center + w/2) * width)
                    y2 = int((y_center + h/2) * height)

                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    label_text = classes[class_id] if classes else f"Class {class_id}"
                    cv2.putText(image, label_text, (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        plt.subplot(2, 2, i + 1)
        plt.imshow(image)
        plt.title(img_name)
        plt.axis('off')

    plt.tight_layout()
    plt.show()

visualize_yolo_data("dataset_generation/merged_dataset", num_samples=4)
import os
import shutil
import random

def split_yolo_dataset(source_dir, output_dir, train_ratio=0.8):
    src_images = os.path.join(source_dir, 'images')
    src_labels = os.path.join(source_dir, 'labels')

    for split in ['train', 'val']:
        os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, 'labels'), exist_ok=True)

    all_images = [f for f in os.listdir(src_images) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    random.shuffle(all_images)

    split_index = int(len(all_images) * train_ratio)
    
    train_files = all_images[:split_index]
    val_files = all_images[split_index:]

    def copy_files(file_list, split_name):
        print(f"Copying {len(file_list)} files to {split_name}...")
        for filename in file_list:
            img_src_path = os.path.join(src_images, filename)
            img_dst_path = os.path.join(output_dir, split_name, 'images', filename)
            
            label_filename = os.path.splitext(filename)[0] + '.txt'
            lbl_src_path = os.path.join(src_labels, label_filename)
            lbl_dst_path = os.path.join(output_dir, split_name, 'labels', label_filename)

            shutil.copy2(img_src_path, img_dst_path)
            if os.path.exists(lbl_src_path):
                shutil.copy2(lbl_src_path, lbl_dst_path)

    copy_files(train_files, 'train')
    copy_files(val_files, 'val')

    if os.path.exists(os.path.join(source_dir, 'classes.txt')):
        shutil.copy2(os.path.join(source_dir, 'classes.txt'), os.path.join(output_dir, 'classes.txt'))

    print("\nDone! Your dataset is split and ready for training.")
    print(f"Total: {len(all_images)} | Train: {len(train_files)} | Val: {len(val_files)}")

source = "dataset_generation/merged_dataset" 
output = "dataset" 
split_yolo_dataset(source, output)
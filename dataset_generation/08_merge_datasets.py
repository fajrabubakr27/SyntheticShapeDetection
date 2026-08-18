import os
import shutil

def merge_yolo_datasets(dir1, dir2, output_dir):
    # إنشاء الفولدرات الجديدة
    sub_folders = ['images', 'labels']
    for sub in sub_folders:
        os.makedirs(os.path.join(output_dir, sub), exist_ok=True)

    def process_folder(source_path, prefix):
        img_source = os.path.join(source_path, 'images')
        lbl_source = os.path.join(source_path, 'labels')

        # التأكد من وجود الفولدرات
        if not os.path.exists(img_source) or not os.path.exists(lbl_source):
            print(f"Error: Could not find images or labels in {source_path}")
            return

        # لف على كل الصور في فولدر الصور
        for filename in os.listdir(img_source):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                # اسم الملف بدون الامتداد
                file_basename = os.path.splitext(filename)[0]
                extension = os.path.splitext(filename)[1]

                # الاسم الجديد (مثلاً: ds1_composite_0000.jpg)
                new_filename = f"{prefix}_{file_basename}"

                # 1. نقل الصورة
                old_img_path = os.path.join(img_source, filename)
                new_img_path = os.path.join(output_dir, 'images', new_filename + extension)
                shutil.copy2(old_img_path, new_img_path)

                # 2. نقل ملف الـ Label المقابل له
                old_lbl_path = os.path.join(lbl_source, file_basename + '.txt')
                new_lbl_path = os.path.join(output_dir, 'labels', new_filename + '.txt')
                
                if os.path.exists(old_lbl_path):
                    shutil.copy2(old_lbl_path, new_lbl_path)
                else:
                    print(f"Warning: Label not found for {filename}")

    # معالجة الفولدر الأول بـ prefix مختلف
    print("Processing first dataset...")
    process_folder(dir1, "org")

    # معالجة الفولدر الثاني بـ prefix مختلف
    print("Processing second dataset...")
    process_folder(dir2, "agm")

    print(f"Done! All files merged into: {output_dir}")

# استخدمي المسارات بتاعتك هنا
dataset1 = r"dataset_generation\yolo_dataset"
dataset2 = r"dataset_generation\augmented_final"
output = "merged_dataset"

merge_yolo_datasets(dataset1, dataset2, output)
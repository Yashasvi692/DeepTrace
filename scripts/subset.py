import os
import shutil
import random

def create_subset_dataset(source_dir, target_dir, max_images_per_class=5000):
    os.makedirs(target_dir, exist_ok=True)
    for class_name in ['fake', 'real']:
        source_class_dir = os.path.join(source_dir, class_name)
        target_class_dir = os.path.join(target_dir, class_name)
        os.makedirs(target_class_dir, exist_ok=True)
        
        images = os.listdir(source_class_dir)
        random.shuffle(images)
        selected_images = images[:min(max_images_per_class, len(images))]
        
        for img in selected_images:
            shutil.copy(os.path.join(source_class_dir, img), os.path.join(target_class_dir, img))
        
        print(f"Copied {len(selected_images)} images from {source_class_dir} to {target_class_dir}")

if __name__ == '__main__':
    source_dir = 'data/test'
    target_dir = 'data/testsubset'
    create_subset_dataset(source_dir, target_dir, max_images_per_class=1000)
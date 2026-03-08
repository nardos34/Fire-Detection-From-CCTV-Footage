import os
import shutil
import random

image_folder = "images"
label_folder = "labels"

train_ratio = 0.8

label_files = [f for f in os.listdir(label_folder) if f.endswith(".txt")]

valid_images = [f.replace(".txt", ".jpg") for f in label_files]

valid_images = [img for img in valid_images if os.path.exists(os.path.join(image_folder, img))]

random.shuffle(valid_images)

split_index = int(len(valid_images) * train_ratio)

train_images = valid_images[:split_index]
val_images = valid_images[split_index:]

os.makedirs("images/train", exist_ok=True)
os.makedirs("images/val", exist_ok=True)
os.makedirs("labels/train", exist_ok=True)
os.makedirs("labels/val", exist_ok=True)


for img in train_images:
    label = img.replace(".jpg", ".txt")

    shutil.move(os.path.join(image_folder, img), os.path.join("images/train", img))
    shutil.move(os.path.join(label_folder, label), os.path.join("labels/train", label))


for img in val_images:
    label = img.replace(".jpg", ".txt")

    shutil.move(os.path.join(image_folder, img), os.path.join("images/val", img))
    shutil.move(os.path.join(label_folder, label), os.path.join("labels/val", label))

print("Dataset split complete.")
print(f"Training images: {len(train_images)}")
print(f"Validation images: {len(val_images)}")
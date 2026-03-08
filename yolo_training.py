import os, shutil, random

image_folder = "images"
label_folder = "labels"

train_ratio = 0.8

label_files = [f for f in os.listdir(label_folder) if f.endswith(".txt")]
valid_images = [f.replace(".txt", ".jpg") for f in label_files]

all_images = [f for f in os.listdir(image_folder) if f.endswith(".jpg")]
for img in all_images:
    if img not in valid_images:
        os.remove(os.path.join(image_folder, img))

random.shuffle(valid_images)

split_index = int(len(valid_images) * train_ratio)

train_images = valid_images[:split_index]
val_images = valid_images[split_index:]

os.makedirs("images/train", exist_ok=True)
os.makedirs("images/val", exist_ok=True)
os.makedirs("labels/train", exist_ok=True)
os.makedirs("labels/val", exist_ok=True)

for img in train_images:
    shutil.move(os.path.join(image_folder, img), "images/train/" + img)
    shutil.move(os.path.join(label_folder, img.replace(".jpg", ".txt")), "labels/train/" + img.replace(".jpg", ".txt"))

for img in val_images:
    shutil.move(os.path.join(image_folder, img), "images/val/" + img)
    shutil.move(os.path.join(label_folder, img.replace(".jpg", ".txt")), "labels/val/" + img.replace(".jpg", ".txt"))
import pandas as pd
import os
from PIL import Image

df = pd.read_csv("image_coordinates_final.csv")

image_folder = "images"
label_folder = "labels"

os.makedirs(label_folder, exist_ok=True)

for _, row in df.iterrows():
    img_name = str(row["img_name"]) + ".jpg"
    x_min = row["Xmin"]
    y_min = row["Ymin"]
    x_max = row["Xmax"]
    y_max = row["Ymax"]

    img_path = os.path.join(image_folder, img_name)
    img = Image.open(img_path)
    
    img_width, img_height = img.size

    box_width = x_max - x_min
    box_height = y_max - y_min
    
    x_center = x_min + (box_width / 2)
    y_center = y_min + (box_height / 2)

    x_center /= img_width
    y_center /= img_height
    box_width /= img_width
    box_height /= img_height

    label_file = img_name.replace(".jpg", ".txt")
    label_path = os.path.join(label_folder, label_file)

    with open(label_path, "w") as fp:
        fp.write(f"0 {x_center} {y_center} {box_width} {box_height}")

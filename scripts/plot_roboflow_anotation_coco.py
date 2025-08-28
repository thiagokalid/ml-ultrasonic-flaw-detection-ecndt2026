import json
import cv2
import numpy as np
import matplotlib.pyplot as plt

with open("../data/imgs_anotated/train/_annotations.coco.json", "r") as f:
    coco_data = json.load(f)

# Map image_id -> file_name
id_to_filename = {img["id"]: img["file_name"] for img in coco_data["images"]}

# Pick first annotation
ann = coco_data["annotations"][13]
img_file = id_to_filename[ann["image_id"]]
img = cv2.imread("../data/imgs_anotated/train/" + img_file)

# Draw segmentation
for seg in ann["segmentation"]:
    pts = np.array(seg).reshape(-1, 2).astype(int)
    cv2.polylines(img, [pts], True, (255,0,0), 2)

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()

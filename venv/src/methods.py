import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import matplotlib.pyplot as plt
import numpy as np
import cv2  # opencv
from PIL import Image
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "samples\\train")

data = []
current_id = 0
label_id = {}
categories = []
for root, dirs, files in os.walk(image_dir):
    for filename in files:
        if filename.endswith("png") or filename.endswith("jpg"):
            path = os.path.join(root, filename)
            label = os.path.basename(os.path.dirname(path)).replace(" ", "-").lower()
            if not label in label_id:
                categories.append(label)
                label_id[label] = current_id
                current_id += 1

            id_ = label_id[label]
            pil_image = cv2.imread(path)
            try:
                img = cv2.resize(pil_image, (480, 480), Image.ANTIALIAS)
                file_id = [label, filename]
                data.append([img, file_id])

            except Exception as e:
                pass

with open("pic_data.pickle", 'wb') as f:
    pickle.dump(data, f)



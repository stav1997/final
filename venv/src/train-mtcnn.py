import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from mtcnn.mtcnn import MTCNN
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle
from matplotlib.patches import Rectangle
import time

file_id = 0
detector = MTCNN()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR,"samples\\train")
recognizer = cv2.face.LBPHFaceRecognizer_create()

x_train = [] #numpy arrays of the pictures
y_train = []
current_id = 0
label_id = {}
start = time.time()

for root, dirs, files in os.walk(image_dir):
    for filename in files:
        if filename.endswith("png") or filename.endswith("jpg"):
            path = os.path.join(root, filename)
            # new_path = os.path.join(root, str(file_id)+'_.jpg')
            # os.rename(path, new_path)
            # file_id = file_id + 1
            # label = os.path.basename(os.path.dirname(new_path)).replace(" ", "-").lower()
            label = os.path.basename(os.path.dirname(path)).replace(" ", "-").lower()

            if not label in label_id:
                label_id[label] = current_id
                current_id += 1

            id_ =label_id[label]
            # pil_image = plt.imread(new_path)
            pil_image = plt.imread(path)
            img = cv2.resize(pil_image, (480, 480), Image.ANTIALIAS)
            image_array = np.array(img, "uint8")
            if len(img.shape)>=3:
                faces = detector.detect_faces(image_array)
                if len(faces)>0:
                    x, y, width, height = faces[0]['box']
                    if y>=0:
                        image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        image_array = np.array(image, "uint8")

                        roi = image_array[y:y + height, x:x + width]
                        x_train.append(roi)
                        y_train.append(id_)  # id of the label


with open("label_mtcnn.pickle", 'wb') as f:
    pickle.dump(label_id, f)

recognizer.train(x_train, np.array(y_train))
recognizer.save("mtcnn_train.yml")
stop = time.time()
print("training is complete on "+str(len(x_train))+" pictures!!")
print(f"Training time: {stop - start}s")

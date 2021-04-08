import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from PIL import Image
import numpy as np
import cv2 #opencv
import pickle
import matplotlib.pyplot as plt
# from sklearn import svm
# from sklearn import metrics
# from sklearn import datasets
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR,"samples\\train")
face_cascade = cv2.CascadeClassifier(
        "C:\\Users\\stav\\PycharmProjects\\finalProject\\venv\\src\\cascades\\haarcascades\\haarcascade_frontalface_alt2.xml")
# recognizer = cv2.face_LBPHFaceRecognizer.create()
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
            label = os.path.basename(os.path.dirname(path)).replace(" ", "-").lower()
            if not label in label_id:
                label_id[label] = current_id
                current_id += 1

            id_ =label_id[label]
            pil_image = Image.open(path).convert("L")  # L stands for gray scale image
            img = pil_image.resize((480, 480), Image.ANTIALIAS)
            image_array = np.array(img, "uint8")
            faces = face_cascade.detectMultiScale(image_array, 1.15, 6, minSize=(60,60))

            for (x, y, w, h) in faces:
                roi = image_array[y:y+h, x:x+w]
                # roi_final = cv2.resize(roi, (200, 200))
                x_train.append(roi)
                y_train.append(id_) #id of the label
                

print(label_id)
with open("label_haar.pickle", 'wb') as f:
    pickle.dump(label_id, f)

recognizer.train(x_train, np.array(y_train))
recognizer.save("haar_train.yml")
stop = time.time()

print("training is complete on "+str(len(x_train))+" pictures!!")
print(f"Training time: {stop - start}s")

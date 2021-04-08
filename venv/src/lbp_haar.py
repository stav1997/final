import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg.matfuncs import eps
from scipy.spatial import distance
import pickle
from PIL import Image

face_cascade = cv2.CascadeClassifier(
        "C:\\Users\\stav\\final\\venv\\src\\cascades\\haarcascades\\haarcascade_frontalface_alt2.xml")
# recognizer = cv2.face_LBPHFaceRecognizer.create()
recognizer = cv2.face.LBPHFaceRecognizer_create()

recognizer.read("C:\\Users\\stav\\final\\venv\\src\\haar_train.yml")
labels = {}

with open("C:\\Users\\stav\\final\\venv\\src\\label_haar.pickle", 'rb') as f:
    lables = pickle.load(f) #load the lables dictionary
    lables = {v:k for k, v in lables.items()} #invert the lables dictionary to be id:name pairs instead of name:id pairs

def lbpHaar(path_):

    pil_image = Image.open(path_)
    img = pil_image.resize((480, 480), Image.ANTIALIAS)
    image_array = np.array(img, "uint8")

    gray_img = img.convert("L")  # L stands for gray scale image
    gray_image_array = np.array(gray_img, "uint8")
    faces = face_cascade.detectMultiScale(gray_image_array, 1.15, 6, minSize=(60, 60))

    for (x, y, w, h) in faces:
        roi = gray_image_array[y:y + h, x:x + w]
        img = cv2.rectangle(image_array, (x, y), (x + w, y + h), (0, 255, 0), 1)
        # roi_final = cv2.resize(roi, (200, 200))

        id_, conf = recognizer.predict(roi)# higher value in the conf means it is less similar
        print(conf)
        if 30 <= conf <= 90:
            print(id_)
            print(lables[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = lables[id_]
            color = (0, 0, 255) #white color for the text
            cv2.putText(image_array, name, (x, y), font, int(3), color, int(1), cv2.LINE_AA)
        break
    return image_array

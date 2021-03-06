import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pickle
from mtcnn.mtcnn import MTCNN
from PIL import Image


recognizer_mtcnn = cv2.face_LBPHFaceRecognizer.create()
recognizer_mtcnn.read("C:\\Users\\stav\\final\\venv\\src\\mtcnn_train.yml")

with open("C:\\Users\\stav\\final\\venv\\src\\label_mtcnn.pickle", 'rb') as f:
    labels = pickle.load(f)  # load the labels dictionary
    labels = {v: k for k, v in
              labels.items()}  # invert the labels dictionary to be id:name pairs instead of name:id pairs


def lbpMtcnn(path_):
    detector = MTCNN()
    pil_image = plt.imread(path_)
    img = cv2.resize(pil_image, (480, 480), Image.ANTIALIAS)
    image_array = np.array(img, "uint8")
    if len(img.shape) >= 3:
        # gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # image_array = np.array(gray_img, "uint8")

        faces = detector.detect_faces(image_array)
        if len(faces) > 0:
            x, y, width, height = faces[0]['box']
            if y >= 0:
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gray_image_array = np.array(gray_img, "uint8")

                roi = gray_image_array[y:y + height, x:x + width]
                img = cv2.rectangle(image_array, (x, y), (x + width, y + height), (0, 255, 0), 1)
                id_, conf = recognizer_mtcnn.predict(roi)  # higher value in the conf means it is less similar
                print(conf)
                if 30 <= conf <= 90:
                    print(id_)
                    print(labels[id_])
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    name = labels[id_]
                    color = (0, 0, 255)  # white color for the text
                    cv2.putText(image_array, name, (x, y), font, int(3), color, int(1), cv2.LINE_AA)

    return image_array

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import matplotlib.pyplot as plt
import numpy as np
import cv2 #opencv
from PIL import Image
from mtcnn.mtcnn import MTCNN
import pickle


with open("C:\\Users\\stav\\final\\venv\\src\\label_haar.pickle", 'rb') as f:
    labels = pickle.load(f)  # load the labels dictionary
    labels = {v: k for k, v in
              labels.items()}  # invert the labels dictionary to be id:name pairs instead of name:id pairs

with open("C:\\Users\\stav\\final\\venv\\src\\model.sav", 'rb') as f:
    model = pickle.load(f)


def svmMtcnn(path_):
    detector = MTCNN()
    orb = cv2.ORB_create()
    temp = []
    pil_image = cv2.imread(path_)
    img = cv2.resize(pil_image, (480, 480), Image.ANTIALIAS)
    image_array = np.array(img, "uint8")
    if len(img.shape) >= 3:
        print("here")
        faces = detector.detect_faces(image_array)
        print(faces)

        if len(faces) > 0:
            print("here")

            x, y, width, height = faces[0]['box']
            if y >= 0:
                image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gray_image_array = np.array(image, "uint8")

                roi = gray_image_array[y:y + height, x:x + width]
                img = cv2.rectangle(image_array, (x, y), (x + width, y + height), (0, 255, 0), 1)
                roi = cv2.resize(roi, (480, 480))
                # kp = orb.detect(roi, None)
                # kp, des = orb.compute(roi, kp)
                # roi1 = cv2.drawKeypoints(roi, kp, None, color=(0, 255, 0), flags=0)

                # plt.imshow(hog_image)
                # ax = plt.gca()
                # for key, value in faces[0]['keypoints'].items():
                #     print(faces[0]['keypoints'].items())
                # # create and draw dot
                #     dot = plt.Circle(value, radius=20, color='orange')
                #     ax.add_patch(dot)

                # plt.show()
                roi = roi.flatten()
                roi = roi.reshape(1, -1)

                # image_array = cv2.drawKeypoints(image_array, kp, None, color=(0, 255, 0), flags=0)
                pred = model.predict(roi)
                print(pred[0])
                # if 30 <= conf <= 90:
                #     print(id_)
                #     print(labels[id_])
                #     font = cv2.FONT_HERSHEY_SIMPLEX
                #     name = labels[id_]
                #     color = (0, 0, 255)  # white color for the text
                #     cv2.putText(image_array, name, (x, y), font, int(3), color, int(1), cv2.LINE_AA)

    return image_array

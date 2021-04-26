import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import matplotlib.pyplot as plt
import numpy as np
import cv2 #opencv
from PIL import Image
from mtcnn.mtcnn import MTCNN
import pickle
import time
import random
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC #support vector classifier
from sklearn import metrics
detector = MTCNN()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "samples\\train")

data = []
features = []
labels = []
current_id = 0
label_id = {}
categories = []
start = time.time()
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
                image_array = np.array(img, "uint8")
                if len(img.shape) >= 3:
                    faces = detector.detect_faces(image_array)
                    if len(faces) > 0:
                        x, y, width, height = faces[0]['box']
                        if y >= 0:
                            image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                            image_array = np.array(image, "uint8")

                            roi = image_array[y:y + height, x:x + width]
                            roi = cv2.resize(roi, (480, 480))
                            roi = roi.flatten()
                            print(roi.shape)
                            data.append([roi, label])
            except Exception as e:
                pass


random.shuffle(data)
for feature, label in data:
    features.append(feature)
    labels.append(label)

train_x, test_x, train_y, test_y = train_test_split(features, labels, test_size=0.10)
model = SVC(C=100, kernel='linear', gamma=0.0001)
model.fit(train_x, train_y)
pred = model.predict(test_x)
accuracy = metrics.accuracy_score(test_y, pred)
print("accuracy: ", accuracy)
print("prediction is:", pred[0])
emo = test_x[0].reshape(480, 480)
plt.imshow(emo, cmap='gray')
plt.show()


with open("svm.pickle", 'wb') as f:
    pickle.dump(label_id, f)

with open("model.sav", 'wb') as f:
    pickle.dump(model, f)
stop = time.time()

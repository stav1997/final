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
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC #support vector classifier

from sklearn import metrics
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "samples\\train")
face_cascade = cv2.CascadeClassifier(
        "C:\\Users\\stav\\final\\venv\\src\\cascades\\lbpcascades\\lbpcascade_frontalface.xml")

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
            pil_image = Image.open(path).convert("L")  # L stands for gray scale image
            try:
                img = pil_image.resize((480, 480), Image.ANTIALIAS)
                image_array = np.array(img, "uint8")
                faces = face_cascade.detectMultiScale(image_array, 1.15, 6, minSize=(60, 60))

                for (x, y, w, h) in faces:
                    roi = image_array[y:y + h, x:x + w]
                    roi = cv2.resize(roi, (480, 480))
                    roi = roi.flatten()
                    data.append([roi, label])

            except Exception as e:
                pass

# pickle_in = open('data_svm_lbp.pickle', 'rb')
# data = pickle.load(pickle_in)
# pickle_in.close()

random.shuffle(data)
for feature, label in data:
    features.append(feature)
    labels.append(label)


train_x, test_x, train_y, test_y = train_test_split(features, labels, test_size=0.25)

model = SVC(C=10, kernel='linear', degree=4, gamma=0.00001, decision_function_shape='ovo', class_weight='balanced')
model.fit(train_x, train_y)
pred = model.predict(test_x)
print(metrics.classification_report(test_y, pred))
accuracy = metrics.balanced_accuracy_score(test_y, pred)
# print("accuracy: ", accuracy)
print("prediction is:", pred[0])
emo = test_x[0].reshape(480, 480)
plt.imshow(emo, cmap='gray')
plt.show()

with open("data_svm_lbp.pickle", 'wb') as f:
    pickle.dump(data, f)

with open("svm_lbp.pickle", 'wb') as f:
    pickle.dump(labels, f)

with open("model_svm_lbp.sav", 'wb') as f:
    pickle.dump(model, f)
stop = time.time()
print(f"Training time: {stop - start}s")


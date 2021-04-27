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
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC #support vector classifier

from sklearn import metrics
detector = MTCNN()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "samples\\train")

# data = []
features = []
labels = []
# current_id = 0
# label_id = {}
# categories = []
# start = time.time()
# for root, dirs, files in os.walk(image_dir):
#     for filename in files:
#         if filename.endswith("png") or filename.endswith("jpg"):
#             path = os.path.join(root, filename)
#             label = os.path.basename(os.path.dirname(path)).replace(" ", "-").lower()
#             if not label in label_id:
#                 categories.append(label)
#                 label_id[label] = current_id
#                 current_id += 1
#
#             id_ = label_id[label]
#             pil_image = cv2.imread(path)
#             try:
#                 img = cv2.resize(pil_image, (480, 480), Image.ANTIALIAS)
#                 image_array = np.array(img, "uint8")
#                 if len(img.shape) >= 3:
#                     faces = detector.detect_faces(image_array)
#                     if len(faces) > 0:
#                         x, y, width, height = faces[0]['box']
#                         if y >= 0:
#                             image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#                             image_array = np.array(image, "uint8")
#
#                             roi = image_array[y:y + height, x:x + width]
#                             roi = cv2.resize(roi, (480, 480))
#                             roi = roi.flatten()
#                             data.append([roi, label])
#             except Exception as e:
#                 pass
pickle_in = open('data.pickle', 'rb')
data = pickle.load(pickle_in)
pickle_in.close()

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

with open("data.pickle", 'wb') as f:
    pickle.dump(data, f)

# with open("svm.pickle", 'wb') as f:
#     pickle.dump(labels, f)

with open("model.sav", 'wb') as f:
    pickle.dump(model, f)
stop = time.time()

#
# param_grid = [
#   {'C': [1, 10, 100], 'degree':[3, 4],'gamma': [0.001, 0.0001, 0.00001], 'kernel': ['linear', 'rbf']},
#  ]
# print(list(ParameterGrid(param_grid)))
#
# scores = ['precision', 'recall']
#
# for score in scores:
#     print("# Tuning hyper-parameters for %s" % score)
#     print()
#
#     model = GridSearchCV(
#         SVC(decision_function_shape='ovo', class_weight='balanced'), param_grid=param_grid,n_jobs=-1
#     )
#
#     model.fit(train_x, train_y)
#
#     print("Best parameters set found on development set:")
#     print()
#     print(model.best_params_)
#     print()
#     print("Grid scores on development set:")
#     print()
#     means = model.cv_results_['mean_test_score']
#     stds = model.cv_results_['std_test_score']
#     for mean, std, params in zip(means, stds, model.cv_results_['params']):
#         print("%0.3f (+/-%0.03f) for %r"
#               % (mean, std * 2, params))
#     print()
#
#     print("Detailed classification report:")
#     print()
#     print("The model is trained on the full development set.")
#     print("The scores are computed on the full evaluation set.")
#     print()
#     y_true, y_pred = test_y, model.predict(test_x)
#     print(metrics.classification_report(y_true, y_pred))
#     print()
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import cv2
import matplotlib.pyplot as plt
import numpy as np
from lbp_mtcnn import lbpMtcnn
from lbp_haar import lbpHaar


if __name__ == '__main__':
    path = "C:\\Users\\stav\\Desktop\\final year project\\faces\\kanye.jpg"
    # path = "C:\\Users\\stav\\PycharmProjects\\finalProject\\venv\\src\\samples\\validation\\shock\\download (5).jpg"
    lbp = lbpMtcnn(path)
    # lbp = lbpHaar(path)
    plt.imshow(np.flipud(lbp), origin='lower')

    plt.show()

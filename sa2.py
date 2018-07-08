# Classifying using parameterised learning

import numpy as np
import cv2
labels = ["dog", "cat", "panda"]
np.random.seed(1)
# randomly initialize our weight matrix and bias vector -- in a
# *real* training and classification task, these parameters would
# be *learned* by our model, but for the sake of this example,
# let’s use random values
W = np.random.randn(3, 3072)
b = np.random.randn(3)
img= cv2.imread('dog.png')
img = cv2.resize(img, (32, 32)).flatten()
scores = W.dot(img) + b
for (label, score) in zip(labels, scores):
    print("[INFO] {}: {:.2f}".format(label, score))

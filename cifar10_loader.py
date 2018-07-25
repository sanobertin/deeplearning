import os
import cv2
import numpy as np
from imutils import paths
testing= 'C:\\Users\\Bertin\\PycharmProjects\\DeepLearning\\cifar\\testing' #cifar10 testing directory
training= 'C:\\Users\\Bertin\\PycharmProjects\\DeepLearning\\cifar\\training' #cifar10 training directory

test_paths= list(paths.list_images(testing))
train_paths= list(paths.list_images(training))


def get_test_data(dirs= test_paths):
    images=[]
    labels=[]
    for path in dirs:
        image= cv2.imread(path)
        images.append(image)
        label= path.split('\\')[-1][13]
        labels.append(label)
        print('[INFO] {} Completed!'.format(path.split('\\')[-1]))
    os.system('cls')
    return (np.array(labels), np.array(images))


def get_train_data(dirs=train_paths):
    images=[]
    labels=[]
    for path in dirs:
        image= cv2.imread(path)
        images.append(image)
        label= path.split('\\')[-1][13]
        labels.append(label)
        print('[INFO] {} Completed!'.format(path.split('\\')[-1]))
    os.system('cls')
    return ( np.array(labels), np.array(images))


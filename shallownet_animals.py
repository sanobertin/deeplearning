# import the necessary packages
import os
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imagetoarrayprepocessor import ImageToArrayPreprocessor
from SimplePreprocessor import SimplePreprocessor
from SimpleDatasetLoader import SimpleDatasetLoader
from shallownet import ShallowNet
from keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cifar10_loader
from keras.callbacks import ModelCheckpoint
# construct the argument parser and parse the arguments
#ap = argparse.ArgumentParser()
#ap.add_argument("-d", "--dataset", required=True, help="path to input dataset ")
#args = vars(ap.parse_args())

# grab the list of images that we’ll be describing
print("[INFO] loading images...")
#imagePaths = list(paths.list_images(args["dataset"]))
# initialize the image preprocessors
#sp = SimplePreprocessor(32, 32)
#iap = ImageToArrayPreprocessor()
(train_labels, train_data)= cifar10_loader.get_train_data()
(test_labels, test_data)=cifar10_loader.get_test_data()
#labels= train_labels+ test_labels
#data= train_data+ test_data
print('[INFO] loading complete!!')
# load the dataset from disk then scale the raw pixel intensities
# to the range [0, 1]
#sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
#(data, labels) = sdl.load(imagePaths, verbose=500)
#data = data.astype("float") / 255.0
# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
#(trainX, testX, trainY, testY) = train_test_split(train_data, train_labels, test_size=0.25, random_state=42)

# convert the labels from integers to vectors
#trainY = LabelBinarizer().fit_transform(trainY)
#testY = LabelBinarizer().fit_transform(testY)
train_labels= LabelBinarizer().fit_transform(train_labels)
test_labels= LabelBinarizer().fit_transform(test_labels)

# initialize the optimizer and model
print("[INFO] compiling model...")
opt = SGD(lr=0.005)
model = ShallowNet.build(width=32, height=32, depth=3, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# train the network
print("[INFO] training network...")
#H= model.fit(data, labels, batch_size=32, epochs=10000, verbose=1)
H = model.fit(train_data, train_labels, validation_data=(test_data, test_labels) , batch_size=32, epochs=20, verbose=1)
print("[INFO] training network complete...", end='\n')

print("[INFO] Serializing/Saving model...", end='\n')
model.save('my_model.h5')
print("[INFO] Serializing/Saving model completed!...", end='\n')
# evaluate the network

print("[INFO] evaluating network...")
predictions = model.predict(test_data, batch_size=32)
print(classification_report(test_data.argmax(axis=1), predictions.argmax(axis=1), target_names=test_labels))
# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 20), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 20), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 20), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, 20), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()


import numpy as np
import cv2
import os


class SimpleDatasetLoader:
    def __init__(self, preprocessors=None):
        # store the image processors
        self.preprocessors= preprocessors
        # if the preprocessors are None, initialize them as an empty list
        if preprocessors is None:
            self.preprocessors=[]

    def load(self, imagePaths, verbose=-1):
        # initialize the list of features and label
        data=[]
        labels=[]
        # loop over the input images
        for (i, imagePath) in enumerate(imagePaths):
            image= cv2.imread(imagePath)
            label=imagePath.split(os.path.sep)[-2]
            if self.preprocessors is not None:
                for p in self.preprocessors:
                    image= p.preprocess(image)

            data.append(image)
            labels.append(label)
            if verbose>0 and i>0 and (i+1)% verbose==0:
                print("[INFO] processed {}/{}".format(i + 1, len(imagePaths)))

        return  (np.array(data), np.array(labels))

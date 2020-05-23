# -*- coding: utf-8 -*-
"""
Created on Fri May 22 20:35:34 2020

@author: Haravindan
"""
import numpy as np
import gzip
import cv2 as cv
import os
import pickle
import tensorflow.keras as keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense


train_path = "C:/Users/Haravindan/Downloads/All-Age-Faces_Dataset/aglined_faces/"
np_train_path = "C:/Users/Haravindan/Downloads/All-Age-Faces_Dataset/numpy_files/train.npy.gz"
np_test_path = "C:/Users/Haravindan/Downloads/All-Age-Faces_Dataset/numpy_files/test.npy.gz"
np_train_label_path = "C:/Users/Haravindan/Downloads/All-Age-Faces_Dataset/numpy_files/train_labels.npy"
np_test_label_path = "C:/Users/Haravindan/Downloads/All-Age-Faces_Dataset/numpy_files/test_labels.npy"

def extract_data(file):
    images = []
    labels = []
    with open(file,'r') as f:
        for line in f:
            w = line.split(" ")
            path = train_path + w[0]
            image = cv.imread(path)
            image = cv.resize(image, (45,60))
            images.append(image)
            labels.append(int(w[1].replace("\n","")))

    return images, labels

if __name__ == "__main__":

    if os.path.exists(np_train_path):

        print("\n..........loading dataset from numpy files..........\n")

        with gzip.GzipFile(np_train_path, "r") as f:
            train_data = np.load(f)
        with gzip.GzipFile(np_test_path, "r") as f:
            test_data = np.load(f)
            
        train_labels = np.load(np_train_label_path)
        test_labels = np.load(np_test_label_path)
        
        # show image using cv
        # print(train_labels[0])
        # cv.imshow("", train_data[0])
        # print(train_data[0].shape)
        # cv.waitKey()
        
    else:

        print("\n..........loading dataset from disk..........\n")
        train_data, train_labels = extract_data("train.txt")
        test_data, test_labels = extract_data("val.txt")

        os.makedirs(os.path.dirname(np_train_path), exist_ok=True)

        with gzip.GzipFile(np_train_path, "w") as f:
            np.save(f, train_data)
        with gzip.GzipFile(np_test_path, "w") as f:
            np.save(f, test_data)

        np.save(np_train_label_path, train_labels)
        np.save(np_test_label_path, test_labels)
        

classifier = Sequential()

classifier.add(Convolution2D(32, 3, input_shape = train_data[0].shape, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))

classifier.add(Convolution2D(64, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))

classifier.add(Flatten())
# classifier.add(Dropout(0.2))
classifier.add(Dense(units = 512, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

opt = keras.optimizers.Adam(lr=0.001)

classifier.compile(optimizer = opt, loss = 'binary_crossentropy', metrics = ['accuracy'])

classifier.fit(train_data, train_labels, batch_size = 32, validation_data=(test_data,test_labels),
               epochs = 20, shuffle=True)

classifier.save('trained_cnn.h5')


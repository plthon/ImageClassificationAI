# import the necessary packages
from __future__ import print_function
# from sklearn.cross_validation import train_test_split
import argparse

from keras.layers import Conv2D, Activation, MaxPooling2D, Dropout, Flatten, Dense
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import datasets, model_selection
from skimage import exposure
import numpy as np
# import imutils
import cv2
import matplotlib.pyplot as plt

# load the MNIST digits dataset
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from keras.datasets import cifar10
from tensorflow import keras
from tensorflow_core.python.keras.models import Sequential

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print("No. of training samples = ", x_train.shape[0])
print("No. of validation set samples = ", x_test.shape[0])

num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))


kVals = range(1, 30, 2)
accuracies = []

models = {
    "knn": KNeighborsClassifier(n_neighbors=1),
    "naive_bayes": GaussianNB(),
    "logit": LogisticRegression(solver="lbfgs", multi_class="auto"),  #
    "svm": SVC(kernel="linear"),
    "decision_tree": DecisionTreeClassifier(),
    "random_forest": RandomForestClassifier(n_estimators=100),
    "mlp": MLPClassifier()
}

print("[INFO] using '{}' model XXXX".format("XXX"))
for k in range(1, 30, 2):
    model = models["mlp"]
    # train the k-Nearest Neighbor classifier with the current value of `k`
    # model = KNeighborsClassifier(n_neighbors=k)
    model.fit(x_train, y_train)
    # evaluate the model and update the accuracies list
    score = model.score(x_test, y_test)
    print("k=%d, accuracy=%.2f%%" % (k, score * 100))
    accuracies.append(score)

i = np.argmax(accuracies)
print("k=%d achieved highest accuracy of %.2f%% on validation data" % (kVals[i],
                                                                       accuracies[i] * 100))

model = KNeighborsClassifier(n_neighbors=kVals[i])
model.fit(x_train, y_train)
predictions = model.predict(x_test)

print("EVALUATION ON TESTING DATA")
print(classification_report(y_test, predictions))

print("Confusion matrix")
print(confusion_matrix(y_test, predictions))
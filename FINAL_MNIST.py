# import the necessary packages
from __future__ import print_function

import argparse
import datetime
import time
from mnist import MNIST
import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, hamming_loss, jaccard_score, multilabel_confusion_matrix
from sklearn.metrics import confusion_matrix, zero_one_loss
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

startTime = time.time()

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, default="knn",
                help="type of python machine learning model to use")
args = vars(ap.parse_args())

"""
print("[INFO] Loading data...")
# load the MNIST digits dataset
mnistZ = datasets.load_digits()
# take the MNIST data and construct the training and testing split, using 75% of the data for training and 25% for
# testing
(trainData, testData, trainLabels, testLabels) = train_test_split(np.array(mnistZ.data),
                                                                  mnistZ.target, test_size=0.25, random_state=42)

# now, let's take 10% of the training data and use that for validation
(trainData, valData, trainLabels, valLabels) = train_test_split(trainData, trainLabels,
                                                                test_size=0.1, random_state=84)
"""

# METHOD TWO
print("[INFO] Loading data...")
mnist = MNIST('/home/plthon/PycharmProjects/ImageClassificationAI/dataset/MNIST/')
trainData, trainLabels = mnist.load_training() #60000 samples
testData, testLabels = mnist.load_testing()    #10000 samples

trainData = np.asarray(trainData).astype(np.float32)
trainLabels = np.asarray(trainLabels).astype(np.int32)
testData = np.asarray(testData).astype(np.float32)
testLabels = np.asarray(testLabels).astype(np.int32)
# METHOD TWO

# show the sizes of each data split
print("training data points: {}".format(len(trainLabels)))
# print("validation data points: {}".format(len(valLabels)))
print("testing data points: {}".format(len(testLabels)))

# define the dictionary of models our script can use, where the key
# to the dictionary is the name of the model (supplied via command
# line argument) and the value is the model itself
models = {
    "knn": KNeighborsClassifier(n_neighbors=1),
    "naive_bayes": GaussianNB(),
    "logit": LogisticRegression(solver="lbfgs", multi_class="auto"),
    "svm": SVC(kernel="linear"),
    "decision_tree": DecisionTreeClassifier(),
    "random_forest": RandomForestClassifier(n_estimators=100),
    "mlp": MLPClassifier()
}

# chosenModel = args["model"]
chosenModel = "knn"

print("\n[INFO] Using '{}' model".format(chosenModel))
fitTime = time.time()
model = models[chosenModel]
model.fit(trainData, trainLabels)
print("Time used (seconds):", datetime.timedelta(seconds=time.time() - fitTime))

print("\n[INFO] Evaluating...")
preditTime = time.time()
predictions = model.predict(testData)
print("Time used (seconds):", datetime.timedelta(seconds=time.time() - preditTime))

# predictions2 = model.predict_proba(testData)
# print(predictions[1])

# show a final classification report demonstrating the accuracy of the classifier
# for each of the digits
print("\n--- EVALUATION ON TESTING DATA ---")
print("Classification Report:")
print(classification_report(testLabels, predictions))
print("\nConfusion Matrix:")
print(confusion_matrix(testLabels, predictions))
sns.heatmap(confusion_matrix(testLabels, predictions), annot=True, lw=2, cbar=False)
plt.ylabel("True Values")
plt.xlabel("Predicted Values")
plt.title("CONFUSSION MATRIX VISUALIZATION")
plt.show()
print("\nZero One Loss:", zero_one_loss(testLabels, predictions, normalize=False))
print("\nHamming Loss:", hamming_loss(testLabels, predictions))
print("\nJaccard Score:", jaccard_score(testLabels, predictions, average=None))
print("\nMultiLabel Confusion Matrix:")
print(multilabel_confusion_matrix(testLabels, predictions))
# Average Precision Score
# Log Loss
# Roc Auc Score
# Coverage Error

"""
print()
# loop over a few random digits
for i in np.random.randint(0, high=len(testLabels), size=(5,)):
    # grab the image and classify it
    image = testData[i]
    prediction = model.predict([image])[0]
    # show the prediction
    imgData = np.array(image, dtype='float')
    pixels = imgData.reshape((8, 8))
    plt.imshow(pixels, cmap='gray')
    plt.annotate(prediction, (3, 3), bbox={'facecolor': 'white'}, fontsize=16)
    print("I think tha digit is : {}".format(prediction))
    plt.show()
    cv2.waitKey(0)
"""
print("\n\n\nTotal Time Taken:", datetime.timedelta(seconds=time.time() - startTime))

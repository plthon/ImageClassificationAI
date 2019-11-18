import argparse
import datetime
import os
import platform
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from six.moves import cPickle as pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, zero_one_loss, hamming_loss, jaccard_score, \
    multilabel_confusion_matrix, log_loss
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

startTime = time.time()

# --- You wont use this. --- Redundant codes. ---
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, default="knn",
                help="type of python machine learning model to use")
args = vars(ap.parse_args())
# ---------------------------------------------------------------------------------------------------------------------

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

img_rows, img_cols = 32, 32
input_shape = (img_rows, img_cols, 3)


def load_pickle(f):
    version = platform.python_version_tuple()
    if version[0] == '2':
        return pickle.load(f)
    elif version[0] == '3':
        return pickle.load(f, encoding='latin1')
    raise ValueError("invalid python version: {}".format(version))


def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb') as f:
        datadict = load_pickle(f)
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3072)
        Y = np.array(Y)
        return X, Y


def load_CIFAR10(ROOT):
    """ load all of cifar """
    xs = []
    ys = []
    for b in range(1, 6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b,))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    return Xtr, Ytr, Xte, Yte


def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=10000):
    # Load the raw CIFAR-10 data
    # TODO: Change the path to appropriate path on your device
    cifar10_dir = '/home/plthon/PycharmProjects/ImageClassificationAI/dataset/cifar-10-batches-py/'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # Subsample the data
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    x_train = X_train.astype('float32')
    x_test = X_test.astype('float32')

    x_train /= 255
    x_test /= 255

    return x_train, y_train, X_val, y_val, x_test, y_test

# --- DATA PREPARATION & PREPROCESSING --------------------------------------------------------------------------------
print("[INFO] Loading data...")
loadTime = time.time()
# Invoke the above function to get our data.
x_train, y_train, x_val, y_val, x_test, y_test = get_CIFAR10_data()

print('Train data shape: ', x_train.shape)
print('Train labels shape: ', y_train.shape)
print('Validation data shape: ', x_val.shape)
print('Validation labels shape: ', y_val.shape)
print('Test data shape: ', x_test.shape)
print('Test labels shape: ', y_test.shape)
print("Time used (seconds):", datetime.timedelta(seconds=time.time() - loadTime))

plt.hist(y_train, bins=30)  # density
plt.ylabel('Frequency')
plt.show()

"""
# visualizing train sample
temp = x_test[91]
# Since every row represents one example to re-map it to image we have to form three 32,32 matrix,
# representing RGB values
R = temp[0:1024].reshape(32, 32)
G = np.reshape(temp[1024:2048], newshape=(32, 32))
B = np.reshape(temp[2048:], newshape=(32, 32))
temp = np.dstack((R, G, B))  # for stacking all these 32,32 matrices.
plt.imshow(temp)
plt.show()
"""
# ---------------------------------------------------------------------------------------------------------------------

# --- TRAINING --------------------------------------------------------------------------------------------------------
models = {
    "knn": KNeighborsClassifier(n_neighbors=1),
    "naive_bayes": GaussianNB(),
    "logit": LogisticRegression(solver="lbfgs", multi_class="auto", max_iter=4000),  #
    "svm": SVC(kernel="poly", gamma="scale", probability=True),
    "decision_tree": DecisionTreeClassifier(),
    "random_forest": RandomForestClassifier(n_estimators=100),
    "mlp": MLPClassifier()
}

# chosenModel = args["model"]
chosenModel = "knn"

print("\n[INFO] Using '{}' model".format(chosenModel))
fitTime = time.time()
model = models[chosenModel]
model.fit(x_train, y_train)
print("Time used (seconds):", datetime.timedelta(seconds=time.time() - fitTime))
# ---------------------------------------------------------------------------------------------------------------------

# --- Evaluation ------------------------------------------------------------------------------------------------------
print("\n[INFO] Evaluating...")
preditTime = time.time()
predictions = model.predict(x_test)
predictions2 = model.predict_proba(x_test)
print("Time used (seconds):", datetime.timedelta(seconds=time.time() - preditTime))

# show a final classification report demonstrating the accuracy of the classifier
# for each of the digits
print("\n--- EVALUATION ON TESTING DATA ---")
print("Classification Report:")
print(classification_report(y_test, predictions))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))
sns.heatmap(confusion_matrix(y_test, predictions), annot=True, lw=2, cbar=False)
plt.ylabel("True Values")
plt.xlabel("Predicted Values")
plt.title("CONFUSSION MATRIX VISUALIZATION")
plt.show()
print("\nZero One Loss:", zero_one_loss(y_test, predictions, normalize=False))
print("\nHamming Loss:", hamming_loss(y_test, predictions))
print("\nJaccard Score:", jaccard_score(y_test, predictions, average=None))
print("\nMultiLabel Confusion Matrix:")
print(multilabel_confusion_matrix(y_test, predictions))
# Average Precision Score
print("\nLog Loss:", log_loss(y_test, predictions2))
# Roc Auc Score
# Coverage Error

"""
print()
# loop over a few random digits
for i in np.random.randint(0, high=len(y_test), size=(5,)):
    # grab the image and classify it
    image = x_test[i]
    prediction = model.predict([image])[0]
    # show the prediction
    imgData = np.array(image, dtype='float')
    pixels = imgData.reshape((32, 32))
    plt.imshow(pixels, cmap='gray')
    plt.annotate(prediction, (3, 3), bbox={'facecolor': 'white'}, fontsize=16)
    print("i think tha digit is : {}".format(prediction))
    plt.show()
    cv2.waitKey(0)
"""
print("\n\n\nTotal Time Taken:", datetime.timedelta(seconds=time.time() - startTime))

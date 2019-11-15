import argparse
import datetime
import os
import platform
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from keras.utils import np_utils
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

# dataset path
home = os.path.expanduser('~')
data_path = os.path.join(home, "data/CIFAR-10/")
data_url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

# CIFAR-10 constants
img_size = 32
img_channels = 3
nb_classes = 10
# length of the image after we flatten the image into a 1-D array
img_size_flat = img_size * img_size * img_channels
nb_files_train = 5
images_per_file = 10000
# number of all the images in the training dataset
nb_images_train = nb_files_train * images_per_file


def load_data(file_name):
    file_path = '/home/plthon/PycharmProjects/ImageClassificationAI/dataset/cifar-10-batches-py/' + file_name

    print('Loading ' + file_name)
    with open(file_path, mode='rb') as file:
        data = pickle.load(file, encoding='bytes')
    raw_images = data[b'data']
    cls = np.array(data[b'labels'])

    images = raw_images.reshape([-1, img_channels, img_size, img_size])
    # move the channel dimension to the last
    images = np.rollaxis(images, 1, 4)

    return images, cls


def load_training_data():
    # pre-allocate the arrays for the images and class-numbers for efficiency.
    images = np.zeros(shape=[nb_images_train, img_size, img_size, img_channels],
                      dtype=int)
    cls = np.zeros(shape=[nb_images_train], dtype=int)

    begin = 0
    for i in range(nb_files_train):
        images_batch, cls_batch = load_data(file_name="data_batch_" + str(i + 1))
        num_images = len(images_batch)
        end = begin + num_images
        images[begin:end, :] = images_batch
        cls[begin:end] = cls_batch
        begin = end

    return images, np_utils.to_categorical(cls, nb_classes)


def load_test_data():
    images, cls = load_data(file_name="test_batch")

    return images, np_utils.to_categorical(cls, nb_classes)


def load_cifar():
    X_train, Y_train = load_training_data()
    X_test, Y_test = load_test_data()

    return X_train, Y_train, X_test, Y_test


# --- DATA PREPARATION & PREPROCESSING --------------------------------------------------------------------------------
print("[INFO] Loading data...")
loadTime = time.time()
# Invoke the above function to get our data.
x_train, y_train, x_test, y_test = load_cifar()

print('Train data shape: ', x_train.shape)
print('Train labels shape: ', y_train.shape)
# print('Validation data shape: ', x_val.shape)
# print('Validation labels shape: ', y_val.shape)
print('Test data shape: ', x_test.shape)
print('Test labels shape: ', y_test.shape)
print("Time used (seconds):", datetime.timedelta(seconds=time.time() - loadTime))

def grayscale(data, dtype='float32'):
    # luma coding weighted average in video systems
    r, g, b = np.asarray(.3, dtype=dtype), np.asarray(.59, dtype=dtype), np.asarray(.11, dtype=dtype)
    rst = r * data[:, :, :, 0] + g * data[:, :, :, 1] + b * data[:, :, :, 2]
    # add channel dimension
    rst = np.expand_dims(rst, axis=3)
    return rst

X_train_gray = grayscale(x_train)
X_test_gray = grayscale(x_test)

# now we have only one channel in the images
img_channels = 1

# plot a randomly chosen image
img = 64
plt.figure(figsize=(4, 2))
plt.subplot(1, 2, 1)
plt.imshow(x_train[img], interpolation='none')
plt.subplot(1, 2, 2)
plt.imshow(X_train_gray[img, :, :, 0], cmap=plt.get_cmap('gray'), interpolation='none')
plt.show()
# ---------------------------------------------------------------------------------------------------------------------

# --- TRAINING --------------------------------------------------------------------------------------------------------
models = {
    "knn": KNeighborsClassifier(n_neighbors=1),
    "naive_bayes": GaussianNB(),
    "logit": LogisticRegression(solver="lbfgs", multi_class="auto"),  #
    "svm": SVC(kernel="poly"),
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

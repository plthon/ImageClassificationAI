import numpy as np
import os
import pickle
import matplotlib.pyplot as plt


def loadData(path):
    listOfTestFiles = os.listdir(path=path)
    train = []
    train_labels = []
    test = []
    test_labels = []

    print("Training files = ", listOfTestFiles[1:6])
    # For collecting Training data:
    for file in listOfTestFiles[1:6]:
        with open(path + file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
            train.append(dict[b'data'])
            train_labels.append(dict[b'labels'])

    print(listOfTestFiles[7])
    # for collecting Testing data
    with open(path + listOfTestFiles[7], 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
        test.append(dict[b'data'])
        test_labels.append(dict[b'labels'])

    dictData = {'train_data': np.reshape(np.array(train), newshape=(
        np.array(train).shape[0] * np.array(train).shape[1], np.array(train).shape[2])),
                'train_labels': np.reshape(np.array(train_labels),
                                           newshape=(np.array(train_labels).shape[0] * np.array(train_labels).shape[
                                               1])), 'test_data': np.reshape(np.array(test), newshape=(
            np.array(test).shape[0] * np.array(test).shape[1], np.array(test).shape[2])),
                'test_labels': np.reshape(np.array(test_labels),
                                          newshape=(np.array(test_labels).shape[0] * np.array(test_labels).shape[1]))}
    return dictData


dataset = loadData(path='/home/plthon/PycharmProjects/ImageClassificationAI/dataset/cifar-10-batches-py/')
print(dataset['train_data'].shape)

# visualizing train sample
temp = dataset['test_data'][99]

# Since every row represents one example to re-map it to image we have to form three 32,32 matrix,
# representing RGB values

R = temp[0:1024].reshape(32, 32)
G = np.reshape(temp[1024:2048], newshape=(32, 32))
B = np.reshape(temp[2048:], newshape=(32, 32))
temp = np.dstack((R, G, B))  # for stacking all these 32,32 matrices.
plt.imshow(temp)
plt.show()

x_train, y_train, x_test, y_test = dataset['train_data'], dataset['train_labels'], dataset['test_data'], dataset[
    'test_labels']

# Splitting the data into train and validation set
# train = 49000 samples and validation set = 1000 samples

train_x, train_y = x_train[0:49000], y_train[0:49000]
val_x, val_y = x_train[49000:], y_train[49000:]

print("No. of training samples = ", train_x.shape[0])
print("No. of validation set samples = ", val_x.shape[0])

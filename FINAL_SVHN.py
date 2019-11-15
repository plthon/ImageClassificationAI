import argparse
import datetime
import time

import matplotlib.pyplot as plt
import scipy.io
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, zero_one_loss, hamming_loss, \
    jaccard_score, multilabel_confusion_matrix, log_loss
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle

startTime = time.time()

# --- You wont use this. --- Redundant codes. ---
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, default="knn",
                help="type of python machine learning model to use")
args = vars(ap.parse_args())
# ---------------------------------------------------------------------------------------------------------------------

# --- DATA PREPARATION & PREPROCESSING --------------------------------------------------------------------------------
print("[INFO] Loading data...")
loadTime = time.time()
# load our dataset
# dataset link: http://ufldl.stanford.edu/housenumbers/
train_data = scipy.io.loadmat('dataset/train_32x32.mat')
# extract the images and labels from the dictionary object
X = train_data['X']
y = train_data['y']
# view an image (e.g. 25) and print its corresponding label
img_index = 9
plt.imshow(X[:, :, :, img_index])
plt.show()
print(y[img_index])

X = X.reshape(X.shape[0] * X.shape[1] * X.shape[2], X.shape[3]).T
y = y.reshape(y.shape[0], )
X, y = shuffle(X, y, random_state=42)
print(X.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=500, test_size=100, random_state=42)

print('Train data shape: ', X_train.shape)
print('Train labels shape: ', y_train.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)
print("Time used (seconds):", datetime.timedelta(seconds=time.time() - loadTime))
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
model.fit(X_train, y_train)
print("Time used (seconds):", datetime.timedelta(seconds=time.time() - fitTime))
# ---------------------------------------------------------------------------------------------------------------------

# --- Evaluation ------------------------------------------------------------------------------------------------------
print("\n[INFO] Evaluating...")
preditTime = time.time()
predictions = model.predict(X_test)
predictions2 = model.predict_proba(X_test)
print("Time used (seconds):", datetime.timedelta(seconds=time.time() - preditTime))

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
    image = X_test[i]
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

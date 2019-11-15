from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def confusion_matrix_with_visual(testLabels, predictions):
    print(confusion_matrix(testLabels, predictions))
    sns.heatmap(confusion_matrix(testLabels, predictions), annot=True, lw=2, cbar=False)
    plt.ylabel("True Values")
    plt.xlabel("Predicted Values")
    plt.title("CONFUSSION MATRIX VISUALIZATION")
    plt.show()
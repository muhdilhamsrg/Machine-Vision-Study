#import numpy as np
from mlxtend.data import loadlocal_mnist
from sklearn import datasets
#from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score
from skimage.feature import hog
from sklearn.preprocessing import LabelBinarizer
from matplotlib import pyplot as plt
from mlxtend.plotting import plot_confusion_matrix
import random


# Load MNIST image dataset
train_images, train_labels = loadlocal_mnist(images_path='./images/mnist-dataset/train-images.idx3-ubyte',
                                             labels_path='./images/mnist-dataset/train-labels.idx1-ubyte')
test_images, test_labels = loadlocal_mnist(images_path='./images/mnist-dataset/t10k-images.idx3-ubyte',
                                             labels_path='./images/mnist-dataset/t10k-labels.idx1-ubyte')

input_image = random.randint(0,100) #random number for select index image

# Preprocessing using HOG Feature Extraction
feature, hog_img= hog(train_images[input_image].reshape(28,28), orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2,2), 
                      visualize=True, block_norm='L2') # extract train image feature
n_dims = feature.shape[0] #dimension of feature
n_samples_train = train_images.shape[0]
X_train, y_train = datasets.make_classification(n_samples=n_samples_train, n_features=n_dims) #matrix for input image and label
n_samples_test = test_images.shape[0]
X_test, y_test = datasets.make_classification(n_samples=n_samples_test, n_features=n_dims) 

print("\nfeature shape: ",feature.shape)
print("n_dims: ",n_dims)
print("n_samples_train: ",n_samples_train)
print("n_samples_test: ",n_samples_test)
print("X_train shape: ",X_train.shape)
print("X_test shape: ",X_test.shape)

# HOG Feature Extraction
def hog_features(img_data, label_data, xdata, ydata):
    # Compute HOG feature
    n_samples = img_data.shape[0]
    print("Extract hog each feature > n_sample:", n_samples)
    
    for i in range(n_samples):
        xdata[i], _ = hog(img_data[i].reshape(28, 28), orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2,2), 
                          visualize=True, block_norm='L2')
        ydata[i] = label_data[i]
    
    #lb = LabelBinarizer()
    #lb.fit(ydata)
    #y_one_hot = lb.transform(ydata)
    
    return xdata, ydata #, y_one_hot

train_hog = hog_features(train_images, train_labels, X_train, y_train)
test_hog = hog_features(test_images, test_labels, X_test, y_test)

X_train_hog = train_hog[0]
y_train_hog = train_hog[1]
X_test_hog = test_hog[0]
y_test_hog = test_hog[1]

#print("train_hog: ",train_hog)
#print("test_hog: ",test_hog)


print("\nInput Image, index: [", input_image,"]")
plt.imshow(test_images[input_image].reshape(28,28), cmap='gray')
plt.show()

# Train with SVM classifier
print("\nSVM classifier procces...")
clf = SVC(max_iter=100)
#clf = SVC(gamma=0.001, C=100, max_iter=100)
clf.fit(X_train_hog, y_train)

# Predictions
y_pred = clf.predict(X_test_hog)
print("\ny_pred: ", y_pred)


out = clf.predict(X_test_hog[input_image].reshape(1, n_dims))
#print("\nout: ",out)

# Evaluate performance
conf_matrix = confusion_matrix(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted")
accuracy = accuracy_score(y_test, y_pred)


# Display results
print("\nPredict:", out[0])
print("\n-- Evaluation Performance --\n")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Confusion Matrix:")
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
plot_confusion_matrix(conf_mat=conf_matrix, class_names=class_names)
#print("Confusion Matrix:\n", conf_matrix)

    
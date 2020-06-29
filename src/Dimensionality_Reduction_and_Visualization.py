import numpy as np
import matplotlib as plt
import pandas as pd
from matplotlib.pyplot import imread
import random
import scipy.io
import cv2
from sklearn.preprocessing import StandardScaler

# reading the digits.mat dataset .
mat = scipy.io.loadmat('../digits/digits.mat')
# getting the digits values and corresponing labels.
digits = mat['digits']
labels = mat['labels']
digits_1 = np.concatenate((digits, labels), axis=1)
# shuffling the dataset and distinguishing digits and labels from randomly shuffled. 
random.shuffle(digits_1)
digits = digits_1[:, 0:400]
labels = digits_1[:, 400]
# splitting dataset into 50% training and 50% test samples.
train_data = digits[:2500]
train_labels = labels[:2500]
test_data = digits[2500:]
test_labels = labels[2500:]
# Then scale both training and testing datasets. 
scaler = StandardScaler()
train_data = scaler.fit_transform(train_data)
test_data = scaler.transform(test_data)
#Question 1
#1.1
#Use PCA to obtain a new set of bases (use the training data set, i.e., 2,500 patterns for PCA). 
#Plot the eigenvalues in descending order. How many components (subspace dimension)
#would you choose by just looking at this plot?

#In order to find eiganvalues, first we need to compute coverianca matrix of given dataset.

cov_digits = np.cov(train_data, rowvar=False)
        

#Finding eiganvalues and eiganvectors of training dataset (eiganvectors are found for the next question.)
eig_val_training, eig_vec_training = np.linalg.eig(cov_digits)

# Sorting the eiganvalues and ploting the result in descending order.


#Plot sorted eiganvaleus
plt.pyplot.plot(eig_val_training)
plt.pyplot.ylabel('eiganvalues')
plt.pyplot.xlabel('index')
plt.pyplot.show()

#I chose 11 components, detailed explanation can be found in report. 
components = 11

#1.2 
#Display the sample mean for the whole training data set as an image (using samples for
#all classes together). Also display the bases (eigenvectors) that you chose as images (e.g.,
#like in Figure 1) and discuss the results with respect to your expectations.

#Mean of the whole digits training dataset.
mean = np.mean(train_data, axis = 0)
mean_img = mean.reshape(20,20)
plt.pyplot.imshow(mean_img.T)
plt.pyplot.show()

#In order to find eigenvectors with max eiganvalues, it is needed to find indexes of eiganvalues that I have chosen.
mean_1, eigenVectors = cv2.PCACompute(train_data, mean=None, maxComponents=components)

eigen_faces = [];

# Reshaping eigenvectors to 20,20 matrix to display as image.
for eigen_vector in eigen_vectors:
    eigen_face = eigen_vector.reshape(20,20)
    eigen_faces.append(eigen_face)
    
# Ploting 
for eigen_face in eigen_faces:
    plt.pyplot.imshow(eigen_face.T)
    plt.pyplot.show()


#1.3 
#Choose different subspaces with dimensions between 1 and 200 (choose at least 20 different
#subspace dimensions, the more the better), and project the data (project both the training
#data and the test data using the transformation matrix estimated from the training data)
#onto these subspaces. Train a Gaussian classifier using data in each subspace (do not
#forget to use half of the data for training and the remaining half for testing).

#First, 20 or more different components are selected in this array.    
different_components = np.array([1,2,5,8,10,15,20,25,35,50,60,75,80,90,100,110,120,130,150,160,170,180,200])

# Then, we need to calculate projection for each components
from sklearn.decomposition import PCA

principal_component_train = []
principal_component_test = []
for i in range(0,different_components.size):
    pca = PCA(n_components=different_components[i])
    principal_component_train.append(pca.fit_transform(train_data))
    principal_component_test.append(pca.transform(test_data))
    
#Then it is needed to train a Gaussian Classifier using the data in each subspace.

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

kernel = RBF(1.0)
predict = []
score_train = []
score_test = []

for i in range(0, different_components.size):
    print(i)
    gpc = GaussianProcessClassifier(kernel=kernel, random_state=0).fit(principal_component_train[i], train_labels)
    predict.append(gpc.predict(principal_component_test[i]))
    score_train.append(gpc.score(principal_component_train[i], train_labels))
    score_test.append(gpc.score(principal_component_test[i], test_labels))
    
#1.4
#Plot classification error vs. the number of components used for each subspace, and discuss
#your results. Compute the classification error for both the training set and the test set
#(training is always done using the training set), and provide two plots.
    
plt.pyplot.plot(different_components, score_train)
plt.pyplot.show()
plt.pyplot.plot(different_components, score_test)
plt.pyplot.show()
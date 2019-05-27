# -*- coding: utf-8 -*-
"""
Created on Sun May 26 14:10:49 2019

@author: Siva
"""

import importData
import numpy as np
import matplotlib.pyplot as plt 
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

# import some data to play with
data = importData.data

#create a dataframe with all training data except the target column
X = data.drop(columns=['isSpam'])

#check that the target variable has been removed
#print(X.head())

#separate target values
y = data['isSpam'].values

#split dataset into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)


scaler = StandardScaler()  
scaler.fit(X_train)

X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=1)  
knn.fit(X_train, y_train)

#show first 5 model predictions on the test data
#print(knn.predict(X_test)[0:5])

#show the accuracy of our model on the test data
print(knn.score(X_test, y_test))

y_pred = knn.predict(X_test) 
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred))

error = []

# Calculating error for K values between 1 and 40
for i in range(1, 40):
   knn2 = KNeighborsClassifier(n_neighbors=i)
   knn2.fit(X_train, y_train)
   pred_i = knn2.predict(X_test)
   error.append(np.mean(pred_i != y_test))

plt.figure(figsize=(12, 6))
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',
        markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')

# =============================================================================
# #create a new KNN model
# knn_cv = KNeighborsClassifier(n_neighbors=3)
# #train model with cv of 5 
# cv_scores = cross_val_score(knn_cv, X, y, cv=5)
# #print each cv score (accuracy) and average them
# print(cv_scores)
# print('cv_scores mean:{}'.format(np.mean(cv_scores)))
# 
# #create new a knn model
# knn3 = KNeighborsClassifier()
# #create a dictionary of all values we want to test for n_neighbors
# param_grid = {'n_neighbors': np.arange(1, 40)}
# #use gridsearch to test all values for n_neighbors
# knn_gscv = GridSearchCV(knn3, param_grid, cv=5)
# #fit model to data
# knn_gscv.fit(X, y)
# print(knn_gscv.best_params_)
# print(knn_gscv.best_score_)
# 
# =============================================================================

# =============================================================================
# #Setup arrays to store training and test accuracies
# neighbors = np.arange(1,40)
# train_accuracy =np.empty(len(neighbors))
# test_accuracy = np.empty(len(neighbors))
# 
# for i,k in enumerate(neighbors):
#     #Setup a knn classifier with k neighbors
#     knn = KNeighborsClassifier(n_neighbors=k)
#     
#     #Fit the model
#     knn.fit(X_train, y_train)
#     
#     #Compute accuracy on the training set
#     train_accuracy[i] = knn.score(X_train, y_train)
#     
#     #Compute accuracy on the test set
#     test_accuracy[i] = knn.score(X_test, y_test) 
# #Generate plot
# plt.title('k-NN Varying number of neighbors')
# plt.plot(neighbors, test_accuracy, label='Testing Accuracy')
# plt.plot(neighbors, train_accuracy, label='Training accuracy')
# plt.legend()
# plt.xlabel('Number of neighbors')
# plt.ylabel('Accuracy')
# plt.show()
# =============================================================================

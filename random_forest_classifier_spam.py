# -*- coding: utf-8 -*-
"""
Created on Mon May 27 19:41:12 2019

@author: Siva
"""

import importData

# import some data
data = importData.data


from sklearn.model_selection import train_test_split

X = data.drop(columns=['isSpam'])
#print(X)
y = data['isSpam'].values

target_names = ["non spam","spam"]

features_train, features_test, labels_train, labels_test = train_test_split(X, y, test_size=0.2, random_state=1)

#Import classifier
from sklearn.ensemble import RandomForestClassifier

#Create an instance of the RandomForestClassifier
rfc = RandomForestClassifier()

#Fit our model to the training features and labels
rfc.fit(features_train,labels_train)

RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
max_depth=None, max_features='auto', max_leaf_nodes=None,
min_impurity_decrease=0.0, min_impurity_split=None,
min_samples_leaf=1, min_samples_split=2,
min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
oob_score=False, random_state=None, verbose=0,
warm_start=False)

rfc_predictions = rfc.predict(features_test)

#print(rfc_predictions)

#Import pandas to create the confusion matrix dataframe
import pandas as pd

#Import classification_report and confusion_matrix to evaluate our model
from sklearn.metrics import classification_report, confusion_matrix

#Create a dataframe with the confusion matrix
confusion_df = pd.DataFrame(confusion_matrix(labels_test, rfc_predictions),
columns=["Predicted " + str(name) for name in target_names],
index = target_names)
print(confusion_df)
print("******************************************")
print(classification_report(labels_test,rfc_predictions))

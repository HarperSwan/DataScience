# -*- coding: utf-8 -*-
"""
Created on Mon May 27 19:41:12 2019

@author: Siva
"""

import importData
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
import matplotlib.pyplot as plt 

# import some data
data = importData.data
X = data.drop(columns=['isSpam'])
y = data['isSpam'].values
features_train, features_test, labels_train, labels_test = train_test_split(X, y, test_size=0.2, random_state=1)
target_names = ["non spam","spam"]

def classicRFC(x):
    #Create an instance of the RandomForestClassifier
    rfc = RandomForestClassifier(n_estimators=x, random_state=0, n_jobs=-1)
    #Fit our model to the training features and labels
    rfc.fit(features_train,labels_train)
    rfc_predictions = rfc.predict(features_test)
    
    
    #Create a dataframe with the confusion matrix
    confusion_df = pd.DataFrame(confusion_matrix(labels_test, rfc_predictions),
    columns=["Predicted " + str(name) for name in target_names],
    index = target_names)
    
    print(confusion_df)
    
    print("******************************************")
    
    print(classification_report(labels_test,rfc_predictions))
    print(accuracy_score(labels_test, rfc_predictions))  
    
    print("*************************************")
    

    

def accuracyByEstimator(x):
    accuracy = []
    n_estimator = []
    # Calculating error for K values between 1 and 40
    for i in range(1, x):
        rfc2 = RandomForestClassifier(n_estimators=i, random_state=0, n_jobs=-1)
        rfc2.fit(features_train,labels_train)
        rfc2_predictions = rfc2.predict(features_test)
        accuracy.append(accuracy_score(labels_test, rfc2_predictions))
        n_estimator.append(i)

    plt.plot(n_estimator, accuracy)
    plt.title('Accuracy / N Estimators')
    plt.xlabel('N Estimators')
    plt.ylabel('Accuracy')

def most_important_features_RFC():
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
    
    clf = RandomForestClassifier(n_estimators=20, random_state=0, n_jobs=-1)
    clf.fit(X_train, y_train)
    
    feat_labels = ['word_freq_make' ,'word_freq_address'
                    ,'word_freq_all' ,'word_freq_3d' ,'word_freq_our'
                    ,'word_freq_over','word_freq_remove','word_freq_internet'
                    ,'word_freq_order','word_freq_mail','word_freq_receive'
                    ,'word_freq_will','word_freq_people','word_freq_report'
                    ,'word_freq_addresses','word_freq_free','word_freq_business'
                    ,'word_freq_email','word_freq_you','word_freq_credit'
                    ,'word_freq_your','word_freq_font','word_freq_000'
                    ,'word_freq_money','word_freq_hp','word_freq_hpl'
                    ,'word_freq_lab'
                    ,'word_freq_labs','word_freq_telnet' ,'word_freq_857'
                    ,'word_freq_data','word_freq_415','word_freq_85'
                    ,'word_freq_technology','word_freq_1999','word_freq_parts'
                    ,'word_freq_pm','word_freq_direct','word_freq_cs'
                    ,'word_freq_meeting','word_freq_original','word_freq_project'
                    ,'word_freq_re','word_freq_edu','word_freq_table'
                    ,'word_freq_conference'
                    ,'char_freq_;','char_freq_('
                    ,'char_freq_[','char_freq_!'
                    ,'char_freq_$','char_freq_#'
                    ,'capital_run_length_average'
                    ,'capital_run_length_longest'
                    ,'capital_run_length_total']
                    
    # Print the name and gini importance of each feature
    #for feature in zip(feat_labels, clf.feature_importances_):
    #    print(feature)
    
    # Create a selector object that will use the random forest classifier to identify
    # features that have an importance of more than 0.15
    sfm = SelectFromModel(clf, threshold=0.05)
    
    # Train the selector
    sfm.fit(X_train, y_train)
    
    ## Print the names of the most important features
    #for feature_list_index in sfm.get_support(indices=True):
    #    print(feat_labels[feature_list_index])
    
    # Transform the data to create a new dataset containing only the most important features
    X_important_train = sfm.transform(X_train)
    X_important_test = sfm.transform(X_test)
    
    # Create a new random forest classifier for the most important features
    clf_important = RandomForestClassifier(n_estimators=20, random_state=0, n_jobs=-1)
    
    # Train the new classifier on the new dataset containing the most important features
    clf_important.fit(X_important_train, y_train)
    # Apply The Full Featured Classifier To The Test Data
    y_pred = clf.predict(X_test)
    
    # View The Accuracy Of Our Full Feature (4 Features) Model
    print(accuracy_score(y_test, y_pred))
    # Apply The Full Featured Classifier To The Test Data
    y_important_pred = clf_important.predict(X_important_test)
    
    # View The Accuracy Of Our Limited Feature (2 Features) Model
    print(accuracy_score(y_test, y_important_pred))



classicRFC(44)
#most_important_features_RFC()
#accuracyByEstimator(50)
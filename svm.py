# -*- coding: utf-8 -*-

import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split 
from sklearn import svm
from sklearn.svm import SVC  
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


data = pd.read_csv("spambase.data", names=['word_freq_make' ,'word_freq_address'
                ,'word_freq_all' ,'word_freq_3d' ,'word_freq_our'
                ,'word_freq_over','word_freq_remove','word_freq_internet'
                ,'word_freq_order','word_freq_mail','word_freq_receive'
                ,'word_freq_will','word_freq_people','word_freq_report'
                ,'word_freq_addresses','word_freq_free','word_freq_business'
                ,'word_freq_email','word_freq_you','word_freq_credit'
                ,'word_freq_your','word_freq_font','word_freq_000'
                ,'word_freq_money','word_freq_hp','word_freq_hpl'
                ,'word_freq_george','word_freq_650','word_freq_lab'
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
                ,'capital_run_length_total'
                ,'isSpam'])

# shuffle of data
# https://stackoverflow.com/questions/29576430/shuffle-dataframe-rows
data = data.sample(frac=1).reset_index(drop=True)

#get var isSpam
# spamdata = data.pop('isSpam').values

#remove variable useless based on documentation :
'''
Our collection of non-spam 
e-mails came from filed work and personal e-mails, and hence
the word 'george' and the area code '650' are indicators of 
non-spam.
'''

data.pop('word_freq_650')

data.pop('word_freq_george')

# import some data
x = data.drop(columns=['isSpam'])
y = data['isSpam'].values
features_train, features_test, labels_train, labels_test = train_test_split(x, y, test_size=0.2, random_state=1)
target_names = ["non spam","spam"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20)  


#svclassifier = SVC(kernel='poly', degree=8) 
svclassifier = SVC(kernel='rbf')  
#svclassifier = SVC(kernel='sigmoid')  


svclassifier.fit(x_train, y_train)  

y_pred = svclassifier.predict(x_test)  
print(confusion_matrix(y_test,y_pred))  
print(classification_report(y_test,y_pred))  
print("accuracy_score = %f" %(accuracy_score(y_test, y_pred)))


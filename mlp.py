import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split 
from sklearn.svm import SVC  
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.neural_network import MLPClassifier 

spamdata = pd.read_csv("~/Desktop/spambase.data")  
#spamdata = pd.read_csv("~/Downloads/bill_authentication.csv")  

print(spamdata.shape)

print(spamdata.head())

# X = spamdata.drop('class', axis=0)  
# y = spamdata['class']  

csvValuesColumnNumber = 57

# x = spamdata.drop('Class', axis=1)  
# y = spamdata['Class']  

x = spamdata.iloc[:, :-1].values 
y = spamdata.iloc[:, csvValuesColumnNumber].values  

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20)  
 
classifier = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)  # mlp -> classifier
classifier.fit(x_train, y_train.ravel())  
y_pred = classifier.predict(x_test)  

print(confusion_matrix(y_test,y_pred))  
print(classification_report(y_test,y_pred))  
print("accuracy_score = %f" %(accuracy_score(y_test, y_pred)))
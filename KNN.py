import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd 
import time

def KNN() :
    
     # load initial data (mails) 
    csvValuesColumnNumber = 57
    
    # load csv
    csvFilePath = "spambase.data";
    mailDataset = pd.read_csv(csvFilePath, header = None)  # names=names,
    
    # Split columns in 2
    dataFieldsValues = mailDataset.iloc[:, :-1].values  # : signifie "tout" -> :-1 signifie "toutes les colonnes sauf la dernière"
    dataLabels = mailDataset.iloc[:, csvValuesColumnNumber].values  
    
    X_train, X_test, y_train, y_test = train_test_split(dataFieldsValues, dataLabels, test_size=0.2, shuffle=True) # test_size = 1 - train_size
    
    useScaler = False;
    
    
    # Use standard scaler
    if useScaler :
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)  
        X_test = scaler.transform(X_test)
        
    y_predict = predictWith("KNN", X_train, X_test, y_train, y_test)
    
    print(confusion_matrix(y_test, y_predict))
    predictedRatio = np.mean(y_predict != y_test)
    
    print("predictedRatio = %f" %(predictedRatio))
    print(classification_report(y_test, y_predict))     
    
    return;


def predictWith(algoName, X_train, X_test, y_train, y_test) : 
        
    y_predict = None
    errorOccured = False
    
    randSeed = int(time.time() * 10000000000) % 4294967295; # Modulo value of unsigned int : 2^32 - 1
    
    print("predictWith randSeed = " + str(randSeed))
    np.random.seed(randSeed)
    
    startTimeMs = int(time.time() * 1000)
    
    
    if (algoName == "KNN") : # K-Nearest Neighbors
        print("predictWith  " + algoName)
        
        from sklearn.neighbors import KNeighborsClassifier
        classifier = KNeighborsClassifier(n_neighbors=4)  #♪ with 4 neighboors closest
        classifier.fit(X_train, y_train) # X_train_scaled
        y_predict = classifier.predict(X_test) #X_test_scaled
    
    if (errorOccured == False) :
        print(classification_report(y_test, y_predict))   
    
    elapsedTimeMs = int(time.time() * 1000) - startTimeMs
    
    return y_predict#, elapsedTimeMs

KNN()

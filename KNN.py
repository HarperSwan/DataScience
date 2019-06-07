import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd 
import time

def KNN() :
    
     # load initial data (mails) 
    csvValuesColumnNumber = 55
    
    # load csv
    csvFilePath = "spambase.data";
    mailDataset = pd.read_csv("spambase.data", names=['word_freq_make' ,'word_freq_address'
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
    
    # Split columns in 2
    dataFieldsValues = mailDataset.iloc[:, :-1].values  # : signifie "tout" -> :-1 signifie "toutes les colonnes sauf la dernière"
    mailDataset.pop('word_freq_650')
    mailDataset.pop('word_freq_george')
    
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

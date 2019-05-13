import importData

from sklearn.naive_bayes import GaussianNB

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.pipeline import make_pipeline

traindata, testdata = train_test_split(importData.data, test_size=0.5)
spamtraindata, spamtestdata = train_test_split(importData.spamdata, test_size=0.5)

clf = make_pipeline(preprocessing.StandardScaler(), GaussianNB())

clf.fit(traindata, spamtraindata)

voulu = spamtestdata
predit = clf.predict(testdata)

print(clf.score(traindata, spamtraindata))


#!/usr/bin/python

import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()



clf = GaussianNB()
t0 = time()
clf.fit(features_train, labels_train)
print()
print("training time:", round(time() - t0, 3), "s")

t1 = time()
pred =  clf.predict(features_test)
print()
print("prediction time", round(time() - t1, 3), "s")

print()
print('Accuracy = ', accuracy_score(pred, labels_test))

#!/usr/bin/python

"""
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:
    Sara has label 0
    Chris has label 1
"""

import sys
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn import svm

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
#clf = svm.LinearSVC()
#clf2 = svm.SVC(kernel='linear')
#clf3 = svm.SVC(kernel='rbf')
clf3 = svm.SVC(kernel='rbf', C=10000.0)
# to speed up the classifier
features_train = features_train[:len(features_train)/100]
labels_train = labels_train[:len(labels_train)/100]


#clf.fit(features_train, labels_train)
#clf2.fit(features_train, labels_train)
clf3.fit(features_train, labels_train)
#pred = clf.predict(features_test)
#pred2 = clf2.predict(features_test)
pred3 = clf3.predict(features_test)
#acc = sum(float(pred-labels_test))
#print acc
#print accuracy_score(labels_test, pred)
#print accuracy_score(labels_test, pred2)
print accuracy_score(labels_test, pred3)

#########################################################
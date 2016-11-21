#!/usr/bin/python

"""
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
import numpy as np


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
from sklearn import tree, metrics
from PIL import Image
from graphviz import Graph
from sklearn.externals.six import StringIO
import pydot

clf = tree.DecisionTreeClassifier(min_samples_split=40)
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
#print pred==labels_test
acc = metrics.accuracy_score(labels_test,pred)
#acc = sum((pred==labels_test).astype(float))/len(labels_test)
print "accuray: ", np.round(acc, decimals=4)
print "feature length: ", len(features_test[0])
print "feature length2: ",  np.shape(features_test)
#dot_data = StringIO.StringIO()
#dot_data = StringIO()
#dot_data = tree.export_graphviz(clf)
#graph = pydot.graph_from_dot_data(dot_data)
#Image(graph.create_png())
#Image(graph.create_png())




#########################################################

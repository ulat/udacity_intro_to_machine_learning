#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import precision_score, recall_score

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### your code goes here 
# split into train and test data
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

clf = tree.DecisionTreeClassifier()

# build decision tree classifier
# step 1: train classifier on all the data
#clf.fit(features, labels)
# step 2 train clf on training set and validate on test set
clf = clf.fit(features_train, labels_train)
preds = clf.predict(features_test)
nb_pois_predicted = sum(preds[preds==1])
nb_true_positives = [x for x,y in zip(preds, labels_test) if x == 1 and y == 1 ]

# print accuracy
#print "accuracy on total dataset: ", clf.score(features, labels)
print "training accuracy: ", clf.score(features_train, labels_train)
print "testing accuracy : ", clf.score(features_test, labels_test)

# More Evaluation

print "Number of POIs predicted: ", nb_pois_predicted
print "Total people count in testset: ", len(labels_test)
print "True Positives : ", len(nb_true_positives)

# Precision and Recall Score for the classifier
print "Precision Score: ", precision_score(labels_test, preds)
print "Recall Score   : ", recall_score(labels_test, preds)

predictions = [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1]
true_labels = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]






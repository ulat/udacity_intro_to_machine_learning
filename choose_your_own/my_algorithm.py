#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture


features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.show()
################################################################################


### your code here!  name your classifier object clf if you want the
### visualization code (prettyPicture) to show you the decision boundary
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

best_algo = None
best_score = 0.0
best_args = None
best_clf = None

################################################################################
# Train Naive Bayes Classifier
################################################################################
# Gaussian Naive Bayes (GaussianNB)
################################################################################

kwargs = None

clf = GaussianNB()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

if accuracy_score(labels_test, pred) > best_accuracy_score:
	best_accuracy_score = accuracy_score(labels_test, pred)
	best_kwargs = kwargs
	best_algorithm = 'GaussianNB'
	best_clf = clf

################################################################################
# Train Support Vector Classifier
################################################################################
# C-Support Vector Classification.
################################################################################

kwargs = {
	'C' : None,
	'kernel' : None,
	'probability' : None,
	'shrinking' : None
}

for C in np.arange(1.0, 3.5, 0.5):
	for kernel in ('linear', 'poly', 'rbf', 'sigmoid'):
		for probability in (True, False):
			for shrinking in (True, False):
				kwargs['C'] = C
				kwargs['kernel'] = kernel
				kwargs['probability'] = probability
				kwargs['shrinking'] = shrinking

				clf = SVC(**kwargs)
				clf.fit(features_train, labels_train)
				pred = clf.predict(features_test)

				if accuracy_score(labels_test, pred) > best_accuracy_score:
					best_accuracy_score = accuracy_score(labels_test, pred)
					best_kwargs = kwargs
					best_algorithm = 'SVC'
					best_clf = clf

################################################################################
# k-Nearest Neighbors
################################################################################
# Classifier implementing the k-nearest neighbors vote.
################################################################################

kwargs = {
	'n_neighbors' : None,
	'weights' : None,
	'algorithm' : None,
	'p' : None,
    'n_jobs' : -1
}

for n_neighbors in range(1, 11):
	for weights in ('uniform', 'distance'):
		for algorithm in ("auto", "ball_tree", "kd_tree", "brute"):
			for p in (1, 2, 3):
				kwargs['n_neighbors'] = n_neighbors
				kwargs['weights'] = weights
				kwargs['algorithm'] = algorithm
				kwargs['p'] = p

				clf = KNeighborsClassifier(**kwargs)
				clf.fit(features_train, labels_train)
				pred = clf.predict(features_test)

				if accuracy_score(labels_test, pred) > best_accuracy_score:
					best_accuracy_score = accuracy_score(labels_test, pred)
					best_kwargs = kwargs
					best_algorithm = 'KNeighbors'
					best_clf = clf

################################################################################
# Random Forest
################################################################################
# A random forest is a meta estimator that fits a number of decision tree
# classifiers on various sub-samples of the dataset and use averaging to
# improve the predictive accuracy and control over-fitting. The sub-sample size
# is always the same as the original input sample size but the samples are drawn
# with replacement if bootstrap=True (default).
################################################################################

kwargs = {
	'n_estimators' : None,
	'criterion' : None,
	'max_features' : None,
	'max_depth' : None,
	'bootstrap' : None,
	'n_jobs' : -1
}

for n_estimators in range(1, 20):
	for criterion in ("gini", "entropy"):
		for max_features in ("auto", "sqrt", "log2", None):
			for max_depth in tuple(range(1, 11)) + (None,):
				for bootstrap in (True, False):
					kwargs['n_estimators'] = n_estimators
					kwargs['criterion'] = criterion
					kwargs['max_features'] = max_features
					kwargs['max_depth'] = max_depth
					kwargs['bootstrap'] = bootstrap

					clf = RandomForestClassifier(**kwargs)
					clf.fit(features_train, labels_train)
					pred = clf.predict(features_test)

					if accuracy_score(labels_test, pred) > best_accuracy_score:
						best_accuracy_score = accuracy_score(labels_test, pred)
						best_kwargs = kwargs
						best_algorithm = 'RandomForest'
						best_clf = clf

################################################################################
# AdaBoost
################################################################################
# An AdaBoost [1] classifier is a meta-estimator that begins by fitting a
# classifier on the original dataset and then fits additional copies of the
# classifier on the same dataset but where the weights of incorrectly
# classified instances are adjusted such that subsequent classifiers focus
# more on difficult cases.
# This class implements the algorithm known as AdaBoost-SAMME [2].
################################################################################

kwargs = {
	'n_estimators' : None,
	'algorithm' : None,
    'learning_rate' : 0.5
}

for n_estimators in range(40, 70):
	for algorithm in ("SAMME", "SAMME.R"):
        for learning_rate in range(0.5,1.5,0.5):
			kwargs['n_estimators'] = n_estimators
    		kwargs['algorithm'] = algorithm
			kwargs['learning_rate'] = learning_rate

    		clf = AdaBoostClassifier(**kwargs)
    		clf.fit(features_train, labels_train)
    		pred = clf.predict(features_test)

    		if accuracy_score(labels_test, pred) > best_accuracy_score:
    			best_accuracy_score = accuracy_score(labels_test, pred)
    			best_kwargs = kwargs
    			best_algorithm = 'AdaBoost'
    			best_clf = clf
################################################################################
# Summaray
################################################################################
print "Best classifier: ", best_algorithm
print "Parameters:\n", best_kwargs
print "Achieved accuracy:", best_accuracy_score
################################################################################
try:
    for name, clf in clfs.items():
        print name
        prettyPicture(clf, features_test, labels_test)
except NameError:
    #pass
    print "eror"

#!/usr/bin/python

import sys
import pickle

import pandas
import numpy as np
import sklearn
from sklearn.svm import SVC

sys.path.append("../tools/")


from tester import dump_classifier_and_data
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV



########################################################################################################################
### Task 1: Select what features you'll use.
########################################################################################################################
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

features_list = ['poi','salary']  # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

#### Analyzing the dataset - creating the featureslist
features_list = ['poi', 'salary', 'to_messages', 'deferral_payments', 'total_payments',
                 'loan_advances', 'bonus', 'restricted_stock_deferred',
                 'deferred_income', 'total_stock_value', 'expenses', 'from_poi_to_this_person',
                 'exercised_stock_options', 'from_messages', 'other', 'from_this_person_to_poi',
                 'long_term_incentive', 'shared_receipt_with_poi', 'restricted_stock', 'director_fees']

# load data from text-file with poi names

poi_file = open("poi_names.txt", "r")

##### How many data points (people) are contained within the dataset?
print "How many data points (people)?", len(data_dict)

# How many features are there for each person?
nbFeatures = []
featuresWithNaN = {}
totalNbFeatures = 0
for person in data_dict:
    nbFeatures.append(len(person))
    totalNbFeatures += len(person)
    for feature, value in data_dict[person].items():
        if value == 'NaN':
            if featuresWithNaN.has_key(feature):
                featuresWithNaN[feature] += 1
            else:
                featuresWithNaN[feature] = 1

print "total number of features on all persons: ", totalNbFeatures
print "how many features at least (per person): ", min(nbFeatures)
print "how many features at most (per person) : ", max(nbFeatures)

##### How many POIs are there in the E+F dataset?
poisInEmaillist = 0

pois = [key for key in data_dict.keys() if data_dict[key]["poi"] == 1]
print "total count of pois                    : ",  len(pois)
print "percentage pois of total persons       : ", round(float(len(pois))/len(data_dict)*100, 2)
print "features with NaN values:              : ", featuresWithNaN
print "percentage of features with NaN value  : ", round(float(len(featuresWithNaN))/len(nbFeatures)*100, 2)

featureWithMaxNaN = [(featureName, nbOfNaN)
                     for featureName, nbOfNaN
                     in featuresWithNaN.iteritems()
                     if nbOfNaN == max(featuresWithNaN.values())]
print "feature with most NaN values           : ", featureWithMaxNaN[0][0], " - ", featureWithMaxNaN[0][1]


########################################################################################################################
### Task 2: Remove outliers
########################################################################################################################

# Remove the outier which resulted from parsing the pdf document
print "data_dic keys: ", "\n".join(data_dict.keys())
data_dict.pop('TOTAL')
data_dict.pop('THE TRAVEL AGENCY IN THE PARK')


########################################################################################################################
### Task 3: Create new feature(s)
########################################################################################################################
### Store to my_dataset for easy export below.
my_dataset = data_dict

# transform the dataset to pandas dataframe for further processing
df = pandas.DataFrame.from_records(list(data_dict.values()))
print df.head()

# convert NaN- text values to numpy nanvalues
df.replace(to_replace='NaN', value=np.nan, inplace=True)
# print total count of nan-values (should remain equal to the total value of text 'NaN's
print df.isnull().sum()
# print dimensions of dataframe (144 remaining dataset after removing 2 outliers
print df.shape

# removing columns with more than 63 nan-s (63 because the feature 'bonus' shows exactly 63 nan-values.
# I think 'bonus' nevertheless is an important feature
for feature, valueList in df.iteritems():
    if valueList.isnull().sum() > 63:
        df.drop(feature, axis=1, inplace=True)

# Drop email address column
if 'email_address' in list(df.columns.values):
    df.drop('email_address', axis=1, inplace=True)

# calculate imputed dataset. replace nan values with 0
df_imp = df.replace(to_replace=np.nan, value=0)
df_imp = df.fillna(0).copy(deep=True)
df_imp.columns = list(df.columns.values)
print df_imp.isnull().sum()
print df_imp.head()

df_imp.describe()

# create new features: poi_email_ratio, ratio_to_poi, ration_from_poi
poi_email_ratio = (df_imp['from_poi_to_this_person'] + df_imp['from_this_person_to_poi']) / \
                  (df_imp['from_messages'] + df_imp['to_messages'])
to_poi_ratio = df_imp['from_this_person_to_poi'] / df_imp['from_messages']
from_poi_ratio = df_imp['from_poi_to_this_person'] / df_imp['to_messages']

df_imp['poi_ratio'] = pandas.Series(poi_email_ratio) * 1000
df_imp['fraction_to_poi'] = pandas.Series(to_poi_ratio) * 1000
df_imp['fraction_from_poi'] = pandas.Series(from_poi_ratio) * 1000

df_column_list = df_imp.columns.values.tolist()

# Scaling features

scale = sklearn.preprocessing.MinMaxScaler(feature_range=(0, 10000), copy=True)
df_imp['salary_scaled'] = scale.fit_transform(df_imp['salary'])
df_imp.drop('salary', axis=1, inplace=True)
df_imp['bonus_scaled'] = scale.fit_transform(df_imp['bonus'])
df_imp.drop('bonus', axis=1, inplace=True)
df_imp['exercised_stock_options_scaled'] = scale.fit_transform(df_imp['exercised_stock_options'])
df_imp.drop('exercised_stock_options', axis=1, inplace=True)
df_imp['expenses_scaled'] = scale.fit_transform(df_imp['expenses'])
df_imp.drop('expenses', axis=1, inplace=True)
df_imp['other_scaled'] = scale.fit_transform(df_imp['other'])
df_imp.drop('other', axis=1, inplace=True)
df_imp['restricted_stock_scaled'] = scale.fit_transform(df_imp['restricted_stock'])
df_imp.drop('restricted_stock', axis=1, inplace=True)
df_imp['total_payments_scaled'] = scale.fit_transform(df_imp['total_payments'])
df_imp.drop('total_payments', axis=1, inplace=True)
df_imp['total_stock_value_scaled'] = scale.fit_transform(df_imp['total_stock_value'])
df_imp.drop('total_stock_value', axis=1, inplace=True)

### Extract features and labels from dataset for local testing
#data = featureFormat(my_dataset, features_list, sort_keys = True)
#labels, features = targetFeatureSplit(data)

########################################################################################################################
### Task 4: Try a varity of classifiers
########################################################################################################################

### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

labels = df_imp['poi'].copy(deep=True).astype(int).as_matrix()
features = (df_imp.drop('poi', axis=1)).fillna(0).copy(deep=True).as_matrix()

print "Comparing basic Scores of classifiers:"

# Try Gaussian NB
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
scores = sklearn.cross_validation.cross_val_score(clf, features, labels)
print "GaussianNB Score                : ", scores

# Randomized Tree
from sklearn.ensemble import ExtraTreesClassifier
clf = ExtraTreesClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=42)
scores = sklearn.cross_validation.cross_val_score(clf, features, labels)
print "Extremely Randomized Trees Score: ", scores

# SVM
from sklearn.svm import SVC
clf = SVC()
scores = sklearn.cross_validation.cross_val_score(clf, features, labels)
print "Support Vector Machines Score   : ", scores

# AdaBoost
from sklearn.ensemble import AdaBoostClassifier
clf = AdaBoostClassifier(n_estimators=100)
scores = sklearn.cross_validation.cross_val_score(clf, features, labels)
print "AdaBoost Score:                 : ", scores

########################################################################################################################
### Task 5: Tune your classifier to achieve better than .3 precision and recall
########################################################################################################################

### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)
from sklearn import decomposition
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer
f1_scorer = make_scorer(f1_score)




ss = sklearn.cross_validation.StratifiedShuffleSplit(labels, 4, test_size=0.1, random_state=42)

#print "Tuning SVC....."

# Doing PCA within pipeline
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
'''

n_components_list=[3, 4, 5, 6, 7, 8, 13]
C_range = np.logspace(-2, 10, 13)
gammaRange = np.logspace(-9, 3, 13)
#kernelList1 = ('linear', 'poly')
kernelList2 = ('poly', 'rbf', 'sigmoid')
paramGrid = dict(reduce_dim__n_components=n_components_list,
                 clf__gamma=gammaRange,
                 clf__C=C_range,
                 clf__kernel=kernelList2)
clf = SVC()

pca = decomposition.PCA()
estimators = [('reduce_dim', pca), ('clf', clf)]
pipe = Pipeline(estimators)

grid = GridSearchCV(pipe, param_grid=paramGrid, scoring=f1_scorer, cv=ss, n_jobs=-1)
grid.fit(features, labels)
print "Best parameters for SVC are %s with a score of %0.2f" % (grid.best_params_, grid.best_score_)

'''
print "Tuning AdaBoost....."
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier

n_components_list=[3, 4, 5, 6, 7, 8, 13]
n_estimators_list = [5, 10, 30, 40, 50, 100, 150, 200, 250, 300]
learning_rate_list = [0.1, 0.5, 1, 1.5, 2, 2.5]
max_depth_list= [1, 3, 5, 7]

parameters = dict(reduce_dim__n_components=n_components_list,
                  clf__n_estimators=n_estimators_list,
                  clf__learning_rate=learning_rate_list,
                  clf__max_depth=max_depth_list)
#clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=8))

clf = GradientBoostingClassifier()
pca = decomposition.PCA()
estimators = [('reduce_dim', pca), ('clf', clf)]
pipe = Pipeline(estimators)

grid = GridSearchCV(pipe, parameters, scoring=f1_scorer, cv=ss, n_jobs=-1)
grid.fit(features, labels)
print "Best parameters for AdaBoost are %s with a F1 score of %0.2f" % (grid.best_params_, grid.best_score_)


########################################################################################################################
### Task 6: Dump your classifier, dataset, and features_list so anyone can
########################################################################################################################

### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

# Train clf Classifier according to the findings from the fine-tuning phase
clf = grid.best_estimator_

dump_classifier_and_data(clf, my_dataset, features_list)
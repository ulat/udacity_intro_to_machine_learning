 # Udacity - Intro to Machine Learning Final Project

The goal of this machine-learning Problem is to find all the POIs (Persons of Interest) from the Enron Dataset. 
These are the employees from Enron who were involved into fraud.

The data-set contains emails from and to Enron employees as well as personal information about financial data like
salary, bonuses, equities, etc.

## Feature Selection

There are three main categories of features available:

  - **financieal features:** ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees']
  - **email features:** ['to_messages', 'email_address', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'poi', 'shared_receipt_with_poi']
  - **POI labels:** [‘poi’]
  
I start using all the features and reducing them if I can manage any improvement.

### Filtering features with a lot of NaN values
When I run my first analyses on all features I get __146__ different persons with a total count of __2202__ features on 
all the persons. 

Not each feature has values on each person. There are persons with values on only __5__ features up to values on __29__
 different features.
 
Among all persons there are __18__ persons identified as POIs - or about __12%__ of all persons are POIs.

A Lot of the features contain *NaN* values. About __13%__ of all features contain _NaN_ values. The feature with the highest
count on _NaN_ values is __loan_advances__ with a total count of __142__ _NaN_ values.
 
### Filtering outliers ###

There were two outlier removed from the dataset before starting processing the data: 'TOTAL' and 'THE TRAVEL AGENCY IN 
THE PARK'. These datapoints are obviously not names of a person but a headingtext that was included into the dataset 
while paring the pdf-document. This datapoint was manually identified by inspecting the dateset and comparing its 
content with the pdf-document.

### Transforming the dataset into pandas dataframe and further analyses

As many machine learning functions from scikit-learn are capable of handling pandas dataframes directliy, I hava decided
 to convert the dictionary into a dataframe.

I replaced the text 'NaN'-values with numpy nan-values and printed the total sum of nan-values on each feature. I decided
to remove features with more than 63 nan-values because this is about a half of the total dataset. Yet the feature 'bonus' 
shows exactly 63 nan-values. I consider this an important feature and this is why I have chosen 63 as a threshold for
removing features because of a lot of nan-values.


## Creating new Features

Let's analyze the ratio of messages from and to a POI to the total amount of emails sent from each person. Probably would
someone more likely be a POI if he had a lot of correspondence with POIs.

Without feature scaling and modifying the features, AdaBoost and SVC return just F1 scores at about 0.50. Therefore let's
scale all features and do some PCA.

To strengthen the proof of connection between POIs lets add two more features, considering:
  - ratio of emails to pois
  - ratio of emails from pois
  
Scaling all the expenses features raises F1 score of SVC from about .50 to .70 - huge increase but still pretty bad.

## Training different Classifiers

I give Gaussian Naive Bayes, Randomized Tree, SVM and AdaBoost a try. 

## Fine Tuning of Classifiers

I have decided to use SVM and AdaBoost for a detailed fine tuning. SVM allows a lot of parameters to wiggle on. AdaBoost
seems to be very fast and accurate. Both classifiers returned reasonable scores in the first run.

I use GridSearch for testing different values to the parameters of the classifiers.

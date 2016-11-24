#!/usr/bin/python

"""
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000

"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

print "Analyzing the Enron Dataset"

################################################################################
# How many data points (people)?
print "How many data points (people)?", len(enron_data)

################################################################################
# For each person, how many features are available?
nbFeatures = []

totalNbFeatures = 0
for person in enron_data:
	nbFeatures.append(len(person))
	totalNbFeatures += len(person)

print "how many features ", nbFeatures, len(nbFeatures), min(nbFeatures), \
	totalNbFeatures

################################################################################
# How many POIs are there in the E+F dataset?
pois = 0
poisInEmaillist = 0
for key in enron_data.keys():
	if "poi" in enron_data[key].keys():
		if enron_data[key]["poi"] == 1:
			pois += 1
print "nb of pois: ", pois

################################################################################
# How Many Pois are in the Email File?
poi_reader = open('../final_project/poi_names.txt', 'r')
poi_reader.readline() # skip url
poi_reader.readline() # skip blank line

poi_count = 0
for poi in poi_reader:
	poi_count += 1

print "pois in email list: ", poi_count

################################################################################
# What is the total value of the stock belonging to James Prentice?
print "Prentice James Total stock Value", \
	enron_data["PRENTICE JAMES"]["total_stock_value"]

################################################################################
# How many email messages do we have from Wesley Colwell to persons of interest?
print "Total mails von Colwell Wesley to POIs", \
	enron_data["COLWELL WESLEY"]["from_this_person_to_poi"]

################################################################################
# Whats the value of stock options exercised by Jeffrey K Skilling?
print "Valeu of Stock Options by Skilling Jeffrey", \
	enron_data["SKILLING JEFFREY K"]["exercised_stock_options"]

################################################################################
# Among Lay, Skilling and Fastow, who took home the most money?
most_paid = ''
highest_payment = 0

for key in ('LAY KENNETH L', 'FASTOW ANDREW S', 'SKILLING JEFFREY K'):
	if enron_data[key]['total_payments'] > highest_payment:
		highest_payment = enron_data[key]['total_payments']
		most_paid = key

print "Most paid to:", most_paid, "highest payment:", highest_payment

################################################################################
# NotANumber
print(enron_data['SKILLING JEFFREY K'])

################################################################################
# How many folks have a quantified salary?
print "total of persons with quantified salary:", \
	len([key for key in enron_data.keys() if enron_data[key]['salary'] != 'NaN'])
# Total of persons without email
print "total of persons without email: ", \
	len([key for key in enron_data.keys() if enron_data[key]['email_address']!='NaN'])

################################################################################
# How many people have NaN for total_payments? What is the percentage of total?
no_total_payments = len([key for key in enron_data.keys() if enron_data[key]["total_payments"] == 'NaN'])
print "total payments with NaN:", float(no_total_payments)/len(enron_data) * 100

################################################################################
# What percentage of POIs in the data have "NaN" for their total payments?
POIs = [key for key in enron_data.keys() if enron_data[key]['poi'] == True]
number_POIs = len(POIs)
no_total_payments = len([key for key in POIs if enron_data[key]['total_payments'] == 'NaN'])
print "percentage of POIs in the data have \"NaN\" for their total payments", \
	float(no_total_payments)/number_POIs * 100

################################################################################
# If 10 POIs with NaN total_payments were added, what is the new number of people?
# What is the new number of people with NaN total_payments?
print "New total number of datasets in enron_data:", len(enron_data) + 10

print "New total for number of datasets with total_payments == 'NaN':", \
	10 + len([key for key in enron_data.keys() if enron_data[key]['total_payments'] == 'NaN'])

################################################################################
# What is the new number of POIs?
new_nb_pois = 10 + len(POIs)
print "new number of POIs: ", new_nb_pois
# What percentage have NaN for their total_payments?
print "percentage with NaN in total_payments", \
	float(no_total_payments+10)/(number_POIs+10) * 100
print len(POIs)/float(31)

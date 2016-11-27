#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
features = ["salary", "bonus"]

### remove the outlier ("TOTAL" - not a datapoint but a spreadsheet summary line)
data_dict.pop('TOTAL',0)

# We would argue that there’s 4 more outliers to investigate; let's look at a
# couple of them. Two people made bonuses of at least 5 million dollars, and
# a salary of over 1 million dollars; in other words, they made out like bandits.
# What are the names associated with those points?
names_to_remove = [name for name in data_dict.keys() \
    if data_dict[name]['bonus'] != 'NaN' \
    and int(data_dict[name]['bonus']) >= 5000000 \
    and data_dict[name]['salary'] != 'NaN' \
    and int(data_dict[name]['salary']) > 1000000
]
print "names of persons who made more than 5Mio bonus and more than 1Mio salary:", \
    sorted(names_to_remove)


data = featureFormat(data_dict, features)




### your code below
for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()

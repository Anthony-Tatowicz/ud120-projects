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

import sys
sys.path.append("../tools/")
import pickle
import pprint
from feature_format import targetFeatureSplit, featureFormat

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "rb"))
pp = pprint.PrettyPrinter(indent=2)

count = 0
known_emails = 0
salary_count = 0
no_total_payment = 0
poi_no_payment = 0

for key, value in enron_data.items():
    if enron_data[key]["poi"] == True:
        count += 1
        print(key)

        if enron_data[key]['total_payments'] == 'NaN':
            poi_no_payment += 1

    if enron_data[key]['email_address'] != 'NaN':
        known_emails += 1

    if enron_data[key]['salary'] != 'NaN':
        salary_count += 1

    if enron_data[key]['total_payments'] == 'NaN':
        no_total_payment += 1


print()
print('Dataset Size: {}'.format(len(enron_data)))
print('POI Count: {}'.format(count))
print('POI Count (adjusted): {}'.format(count + 10))
print('Email Count: {}'.format(known_emails))
print('Salaries Count: {}'.format(salary_count))
print('No total_payments: {}'.format(no_total_payment))
print('No total_payments (adjusted): {}'.format(no_total_payment + 10))
print('No total_payments percent: {:.2%}'.format(no_total_payment / len(enron_data)))
print("POI's with no total_payments: {:.2%}".format(poi_no_payment / len(enron_data)))
print("POI's with no total_payments (adjusted): {:.2%}".format((poi_no_payment + 10) / (len(enron_data) + 10)))


# print(enron_data["LAY KENNETH L"]['total_payments'])
# print(enron_data["SKILLING JEFFREY K"]['total_payments'])
# print(enron_data["FASTOW ANDREW S"]['total_payments'])

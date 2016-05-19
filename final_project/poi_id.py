#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary','bonus', 'bonus_ratio'] # You will need to use more features


### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
import pprint

my_dataset = data_dict
my_dataset.pop('TOTAL', 0)

'''
Feature ideas:
salary / average
bonus / average <-- Seems to be the a good features to add
from_poi_to_this_person / to_messages
from_this_person_to_poi / from_messages <-X- Due to data limitations
'''

for key, value in my_dataset.items():
    if my_dataset[key]['bonus'] == 'NaN':
        my_dataset.pop(key)


bonuses = [int(my_dataset[key]['bonus']) for key,value in my_dataset.items()]
avg = sum(bonuses) / len(bonuses)

for key, item in my_dataset.items():
    p_bonus = float(my_dataset[key]['bonus']) / avg
    entry = {'bonus_ratio' : p_bonus}
    item.update(entry)


pp = pprint.PrettyPrinter(indent=2)
pp.pprint(my_dataset['METTS MARK'])

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.decomposition import RandomizedPCA
from sklearn.grid_search import GridSearchCV

param_grid = {
        #  'svm__C': [1e3, 5e3, 1e4, 5e4, 1e5],
        #   'svm__gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
          'reduce_dim__n_components' : [10, 20, 30]
          }

estimators = [('reduce_dim', RandomizedPCA(whiten=True)),
              ('NB', GaussianNB())]

clf = Pipeline(estimators)
clf = GridSearchCV(clf, param_grid=param_grid)




### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.1, random_state=42)


clf.fit(features_train, labels_train)
print('\n')
print "Best estimator found by grid search:"
print clf.best_estimator_
pred = clf.predict(features_test)



from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

acc_score = accuracy_score(labels_test, pred)
print('\n')
print('Accuracy Score : {}'.format(acc_score))

print('\n')
p_score = precision_score(labels_test, pred)
print('precision_score : {}'.format(p_score))

print('\n')
r_score = recall_score(labels_test, pred)
print('recall_score : {}'.format(r_score))
print('\n')




### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)

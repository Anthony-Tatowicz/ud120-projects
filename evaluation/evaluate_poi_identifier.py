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

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)


### your code goes here
from sklearn import cross_validation
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
X_train, X_test, y_train, y_test = cross_validation.train_test_split(
        features, labels, test_size=0.3, random_state=42)


### it's all yours from here forward!
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

pred = clf.predict(X_test)

num_pois = len([x for x in pred if x == 1])
print('Number of pois {}'.format(num_pois))

print('Number of test samples {}'.format(len(X_test)))

score = accuracy_score(pred, y_test)
print('Accuracy: {}'.format(score))

true_pos = [x for i,x in enumerate(pred)  if x == y_test[i]]
print(true_pos)

p_score = precision_score(y_test, pred)
print('precision_score : {}'.format(p_score))

r_score = recall_score(y_test, pred)
print('recall_score : {}'.format(r_score))

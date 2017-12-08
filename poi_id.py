#!/usr/bin/python

import sys
import pickle
import os
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest,f_classif


features_list = ['poi','salary','to_messages','total_payments','bonus','restricted_stock','shared_receipt_with_poi','total_stock_value','expenses','other','from_this_person_to_poi','from_poi_to_this_person','mail_content_discussion_threads','mail_content_deleted_items','mail_content_all_documents','mail_content_notes_inbox','mail_content_inbox']


# Load the dictionary containing the dataset

with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# Deleting the outliers

data_dict.pop('TOTAL', None)

# creating the new features from the emails

userMailTypeMap = {}
keyObj = ['discussion_threads','deleted_items','all_documents','notes_inbox','inbox']
for key in data_dict:
    from_file = 'emails_by_address/from_' + data_dict[key]['email_address'] + '.txt'
    to_file = 'emails_by_address/to_' + data_dict[key]['email_address'] + '.txt'
    obj = {}
    if os.path.isfile(from_file):
        with open(from_file) as openfile:
            for line in openfile:
                try:
                    identifier = line.split('/')[3]
                    if identifier in keyObj:
                        obj[identifier]
                        obj[identifier] = obj[identifier] + 1

                except:
                    identifier = line.split('/')[3]
                    if identifier in keyObj:
                        obj[identifier] = 0
    userMailTypeMap[key] = obj


for name in userMailTypeMap:
    for key in keyObj:
        try:
            userMailTypeMap[name][key]
            data_dict[name]['mail_content_'+key] = userMailTypeMap[name][key]
        except:
            data_dict[name]['mail_content_'+key] = 0


# Store to my_dataset for easy export below.

my_dataset = data_dict

# Extract features and labels from dataset for local testing

data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

features = SelectKBest(f_classif,k=12).fit_transform(features, labels)

from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.2, random_state=42)

# Scaling the features using MinMaxScaler

scaler = MinMaxScaler()
scaler = scaler.fit(features_train)
features_train = scaler.transform(features_train)
features_test = scaler.transform(features_test)

estimators = [('reduce_dim', PCA()), ('clf', DecisionTreeClassifier())]
pipe = Pipeline(estimators)
param_grid = dict(reduce_dim__n_components=[3,5,7],clf__min_samples_split=[2,5,10])
clf = GridSearchCV(pipe, param_grid=param_grid)
clf = clf.fit(features_train,labels_train)
print clf.score(features_test,labels_test)


dump_classifier_and_data(clf, my_dataset, features_list)
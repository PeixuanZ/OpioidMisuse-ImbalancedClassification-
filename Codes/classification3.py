# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 11:35:15 2020

@author: Peixuan Z
"""

import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn import model_selection
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from imblearn.ensemble import RUSBoostClassifier
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv('C:\PSU\Courses\Fall2020\IE582\Project\Codes\data2019.csv').astype('category')
data = data.drop(data.columns[[0]], axis=1)  #
data[['K6SCMON']] = data[['K6SCMON']].astype('int64')
data[['K6SCMON']] = (data[['K6SCMON']]-data[['K6SCMON']].min())/(data[['K6SCMON']].max()- data[['K6SCMON']].min())
data[['AGE2_new']] = pd.factorize(data[['AGE2_new']].values.ravel())[0]

# Up-sampling (Oversampling) is the process of randomly duplicating observations from the minority class
from sklearn.utils import resample
# Separate majority and minority classes
data_major = data[data.OPINMYR==0]
data_minor = data[data.OPINMYR==1]
# Upsample minority class
data_minor_upsampled = resample(data_minor, 
                                 replace=True,     # sample with replacement
                                 n_samples=39276,    # to match majority class
                                 random_state=123)
df_upsampled = pd.concat([data_major, data_minor_upsampled])
y = df_upsampled.OPINMYR
X_balanced = df_upsampled.drop('OPINMYR', axis=1)
X = X_balanced
Y = y


skf = StratifiedKFold(n_splits=10)
print("GLM")
# GLM
for train_index, test_index in skf.split(X, Y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
    model = LogisticRegression(penalty='none')
    model.fit(X_train, Y_train)
    y_prob = model.predict_proba(X_test)
    y_pred = (y_prob[:,1]>=0.5)
    fpr, tpr, thresholds = metrics.roc_curve(Y_test, y_pred, pos_label=1)
    print(metrics.auc(fpr, tpr))

print("penalized GLM ")
for train_index, test_index in skf.split(X, Y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
    model = LogisticRegression(penalty = 'l1', solver = 'liblinear')
    model.fit(X_train, Y_train)
    y_prob = model.predict_proba(X_test)
    y_pred = (y_prob[:,1]>=0.5)
    fpr, tpr, thresholds = metrics.roc_curve(Y_test, y_pred, pos_label=1)
    print(metrics.auc(fpr, tpr))


print("Decision Tree")
# Decision Trees
for train_index, test_index in skf.split(X, Y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
    model = DecisionTreeClassifier()
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)
    fpr, tpr, thresholds = metrics.roc_curve(Y_test, y_pred, pos_label=1)
    print(metrics.auc(fpr, tpr))


print("MLP")
# MLP

for train_index, test_index in skf.split(X, Y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
    model = MLPClassifier(random_state=1)
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)
    fpr, tpr, thresholds = metrics.roc_curve(Y_test, y_pred, pos_label=1)
    print(metrics.auc(fpr, tpr))

print("RUSboost")
# RUSboost
for train_index, test_index in skf.split(X, Y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
    model = RUSBoostClassifier(random_state=0)
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)
    fpr, tpr, thresholds = metrics.roc_curve(Y_test, y_pred, pos_label=1)
    print(metrics.auc(fpr, tpr))
    
print("Random Forests")
for train_index, test_index in skf.split(X, Y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
    model = RandomForestClassifier(random_state=0)
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)
    fpr, tpr, thresholds = metrics.roc_curve(Y_test, y_pred, pos_label=1)
    print(metrics.auc(fpr, tpr))
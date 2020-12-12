# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 22:11:43 2020

@author: Peixuan Z
"""
"""
undersampling/Oversampling

GLM SVM RF
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

data[['K6SCMON']] = data[['K6SCMON']].astype('int64')
data[['K6SCMON']] = (data[['K6SCMON']]-data[['K6SCMON']].min())/(data[['K6SCMON']].max()- data[['K6SCMON']].min())
data[['AGE2_new']] = pd.factorize(data[['AGE2_new']].values.ravel())[0]

X = data.drop(['OPINMYR'],axis = 1)
Y = data[['OPINMYR']]

skf = StratifiedKFold(n_splits=10)
print("GLM")
# GLM
for train_index, test_index in skf.split(X, Y):
    X_train, X_test = X.loc[train_index], X.loc[test_index]
    Y_train, Y_test = Y.loc[train_index], Y.loc[test_index]
    model = LogisticRegression(penalty='none')
    model.fit(X_train, Y_train)
    y_prob = model.predict_proba(X_test)
    y_pred = (y_prob[:,1]>=0.135)
    fpr, tpr, thresholds = metrics.roc_curve(Y_test, y_pred, pos_label=1)
    print(metrics.auc(fpr, tpr))

print('penalized GLM')
for train_index, test_index in skf.split(X, Y):
    X_train, X_test = X.loc[train_index], X.loc[test_index]
    Y_train, Y_test = Y.loc[train_index], Y.loc[test_index]
    model = LogisticRegression(penalty = 'l1', solver = 'liblinear')
    model.fit(X_train, Y_train)
    y_prob = model.predict_proba(X_test)
    y_pred = (y_prob[:,1]>=0.135)
    fpr, tpr, thresholds = metrics.roc_curve(Y_test, y_pred, pos_label=1)
    print(metrics.auc(fpr, tpr))

print("Decision Trees")  
# Decision Trees
for train_index, test_index in skf.split(X, Y):
    X_train, X_test = X.loc[train_index], X.loc[test_index]
    Y_train, Y_test = Y.loc[train_index], Y.loc[test_index]
    model = DecisionTreeClassifier()
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)
    fpr, tpr, thresholds = metrics.roc_curve(Y_test, y_pred, pos_label=1)
    print(metrics.auc(fpr, tpr))
    

print("MLP")
# MLP

for train_index, test_index in skf.split(X, Y):
    X_train, X_test = X.loc[train_index], X.loc[test_index]
    Y_train, Y_test = Y.loc[train_index], Y.loc[test_index]
    model = MLPClassifier(random_state=1)
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)
    fpr, tpr, thresholds = metrics.roc_curve(Y_test, y_pred, pos_label=1)
    print(metrics.auc(fpr, tpr))
 
    
print("RUSboost") 
# RUSboost
for train_index, test_index in skf.split(X, Y):
    X_train, X_test = X.loc[train_index], X.loc[test_index]
    Y_train, Y_test = Y.loc[train_index], Y.loc[test_index]
    model = RUSBoostClassifier(random_state=0)
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)
    fpr, tpr, thresholds = metrics.roc_curve(Y_test, y_pred, pos_label=1)
    print(metrics.auc(fpr, tpr))
    
print("Random Forests")
for train_index, test_index in skf.split(X, Y):
    X_train, X_test = X.loc[train_index], X.loc[test_index]
    Y_train, Y_test = Y.loc[train_index], Y.loc[test_index]
    model = RandomForestClassifier(random_state=0)
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)
    fpr, tpr, thresholds = metrics.roc_curve(Y_test, y_pred, pos_label=1)
    print(metrics.auc(fpr, tpr))


# using R
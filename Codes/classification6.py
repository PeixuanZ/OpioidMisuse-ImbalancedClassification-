# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 14:12:02 2020

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
import shap

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

# Split the data into train and test data:
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.8)
# Build the model with the random forest regression algorithm:
model = RandomForestClassifier(random_state=0)
model.fit(X_train, Y_train)
sorted_idx = model.feature_importances_.argsort()
plt.barh(X_train.columns[sorted_idx], model.feature_importances_[sorted_idx])
plt.xlabel("Random Forest Feature Importance")

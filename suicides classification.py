# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 22:05:33 2019

@author: Shubham
"""

#libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#dataset
dataset = pd.read_csv('master.csv')
X = dataset.iloc[:, [3,5,6,10]].values
y = dataset.iloc[:, -1].values

#categorical values
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_x = LabelEncoder()
X[:, 0] = label_x.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features= [0])
X = onehotencoder.fit_transform(X).toarray()
label_y = LabelEncoder()
y = label_y.fit_transform(y)
X = X[:, 1:]

#spliting data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#feature scalling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)

#fitting
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=500,
                                    n_jobs= -1,
                                    criterion='gini')
classifier.fit(X_train,y_train)

#prediction
y_pred = classifier.predict(X_test)

#confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
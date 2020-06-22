# -*- coding: utf-8 -*-
"""
Created on Tue May 26 04:46:02 2020

@author: Rishma Manna
"""
#BUILDING AN ANN

# Classification template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

labelencoder_X_1 = LabelEncoder()
X[:,1] = labelencoder_X_1.fit_transform(X[:,1])

labelencoder_X_2 = LabelEncoder()
X[:,2] = labelencoder_X_2.fit_transform(X[:,2])


ct = ColumnTransformer([("Geography",OneHotEncoder(),[1])], remainder = "passthrough")
X = ct.fit_transform(X)

X = X[:,1:]


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# --------------------- DATA PREPROCESSING COMPLETE ----------------------------

import keras
from keras.models import Sequential
from keras.layers import Dense

#Initializing the ANN

classifier = Sequential()


#Adding the input layer and the first hidden layer with DROPOUT
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))
classifier.add(Dropout(p = 0.1))


# Adding the seocnd hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
classifier.add(Dropout(p = 0.1))

# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
classifier.add(Dropout(p = 0.1))

# Compiling the ANN

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to our training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

# Predicting the Test Set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

new_prediction = classifier.predict(sc.transform(np.array([[0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
new_prediction = (new_prediction > 0.5)


# EVALUATING, IMPROVING AND TUNING THE ANN

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

def build_classifier():
    
    classifier = Sequential()  

    classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))

    classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

    classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    return classifier
    
classifer = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)
accuracies = cross_val_score(estimator = classifer, X = X_train, y = y_train, cv = 10, n_jobs = -1)

mean = accuracies.mean()
variance = accuracies.std()


# TUNING THE ANN USING PARAMETER TUNING AND GRID SEARCH 
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

def build_classifier(optimizer):
    
    classifier = Sequential()  

    classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))

    classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

    classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    return classifier
    
classifer = KerasClassifier(build_fn = build_classifier)

# GRID SEARCH

parameters = {'batch_size' : [25,32], 'epochs' : [100,500], 'optimizer' : ['adam','rmsprop']}

grid_search = GridSearchCV(estimator = classifier, param_grid = parameters, scoring = 'accuracy', cv = 10)


grid_search = grid_search.fit(X_train, y_train)

best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_






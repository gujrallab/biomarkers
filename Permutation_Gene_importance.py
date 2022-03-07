# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 13:37:33 2019

"""

# Importing the libraries
import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense
from statistics import mean

# Matrix processing
matrix = pd.read_csv('final_matrix.csv')
matrix.set_index('Gene')
matrix2 = matrix.values
matrix3 = matrix2.T
final_matrix = pd.DataFrame(matrix3)


# Importing the dataset
X = final_matrix.iloc[1:, 0:26323].values
y = final_matrix.iloc[1:, 26323].values

# Building classifier (Fill with hyperparameters of choice)
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 100, kernel_initializer = 'normal', activation = 'elu', input_dim = len(X[0])))
    classifier.add(Dense(units = 100, kernel_initializer = 'normal', activation = 'elu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'normal' , activation = 'sigmoid')) 
    classifier.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics=[ 'accuracy'])
    return classifier

classifier = build_classifier()   
classifier.fit(X,y, epochs=45, batch_size=4)

# Cross entropy
def cross_entropy(pred, target, epsilon=1e-12):
    pred = np.clip(pred, epsilon, 1. - epsilon)
    N = pred.shape[0]
    ce = -np.sum(target*np.log(pred+1e-9))/N
    return ce

#Gene Importance
results = []

y_pred_base = classifier.predict(X)
y_true = y.tolist()
base_score = cross_entropy(y_pred_base, y_true)
for kinase_number in range(len(X[0])):
    error_increase_list = []
    X_copy = np.copy(X)
    for shuffle_number in range (100):
        #np.random.seed(0)
        np.random.shuffle(X_copy[:,kinase_number])
        y_pred_new = classifier.predict(X_copy)
        new_score = cross_entropy(y_pred_new, y_true)
        error_increase = new_score - base_score
        error_increase_list.append(error_increase)
    mean_error = mean(error_increase_list)
    results.append(mean_error)
    print(kinase_number)
    
results_df = pd.DataFrame(results)
results_df["genes"] = matrix["Gene"]

#Export to excel
results_df.to_excel("gene_importance.xlsx", sheet_name='1')
            




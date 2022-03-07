#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 23:11:15 2020

"""


# Importing the libraries
import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from statistics import mean
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

#Matrix Processing
matrix = pd.read_csv('final_matrix.csv')
matrix.set_index('Gene')
matrix2 = matrix.values
matrix3 = matrix2.T
final_matrix = pd.DataFrame(matrix3)

#Gene List
gene_list = pd.read_csv('recursive_elimination_genes_biomarkerID.csv') # Fill with Genes of Interest
gene_list = gene_list.values.tolist()

genes = []
for gene in gene_list:
    genes.append(gene[0])

# Formatting the dataset
X = final_matrix.iloc[1:, 0:26323]
X.columns = final_matrix.iloc[0, 0:26323].values.tolist()
X2 = X[genes].values
y2 = final_matrix.iloc[1:, 26323]


#Building Classifier
def build_classifier(optimizer, init, activation, hl, drop_percent, num_hidden_layers):
    classifier = Sequential()
    classifier.add(Dense(units = hl, kernel_initializer = init, activation = activation, input_dim = len(genes)))
    classifier.add(Dropout(drop_percent))
    for num in range(num_hidden_layers - 1):
        classifier.add(Dense(units = hl, kernel_initializer = init, activation = activation))
        classifier.add(Dropout(drop_percent))
    classifier.add(Dense(units = 1, kernel_initializer = init, activation = 'sigmoid'))
    classifier.compile(loss = 'binary_crossentropy', optimizer= optimizer, metrics=[ 'accuracy'])
    return classifier
model = KerasClassifier(build_fn=build_classifier)

# Cross validation
param_grid = {'batch_size': [4], #FILL 
              'init': ['normal'], #FILL
              'epochs': [45], #FILL
              'activation': ['elu'],#FILL
              'optimizer': ['adam'], #FILL
              'num_hidden_layers': [2], #FILL
              'drop_percent': [0.0], #FILL
              'hl': [100]} #FILL
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs= 1, cv=19)
grid_result = grid.fit(X2, y2)

# Cross validation Score

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']


for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
    
print(means)
    
best_parameters = grid_result.best_params_
best_accuracy = grid_result.best_score_

for k, v in best_parameters.items():
    print(k, v)
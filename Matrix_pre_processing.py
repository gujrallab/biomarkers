#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 10:26:06 2020

"""


# Importing the libraries
import numpy as np
import pandas as pd

#Extracting Pre Data
dataset = pd.read_csv('matrix.csv')
col_names = []

for col_name in dataset.columns:
    if 'Pre' in col_name:
        col_names.append(col_name)
        
dataset2 = dataset[col_names]
matrix = pd.DataFrame({'num': range(26324)})

for num in [1,2,3,4,6,7,8,12,15,20,24,25,26,27,28,29,31,33,35]:
    col_names_all = []
    for col_name2 in dataset.columns:
        if 'Pre_P' + str(num) + "." in col_name2:
            col_names_all.append(col_name2)
    dataset_shortened = dataset2[col_names_all]
    mean_value = dataset_shortened.mean(axis = 1)
    matrix['Pre_P' + str(num)] = mean_value.values
    
#Exporting to file
matrix.to_csv("final_matrix.csv")

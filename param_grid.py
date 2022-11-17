# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 08:50:53 2021

@author: Edson cilos
"""

#Standard Packages 
import numpy as np
from scipy.stats import loguniform

#Sklearn API

from sklearn.svm import SVC
from sklearn.model_selection import ParameterSampler

#Config file
import config

#Fix seed to reproducibility
seed = config._seed()

#Basic grid structure for classical algorithms, expecpt neural network
def classical_grid():

    #Suppport vector machine with linear kernel
    svc_1 = {
        'estimator': [
            SVC(
                kernel = 'linear'
                , probability=True
                , random_state = seed
            )
        ]
        , 'estimator__C' : loguniform(1e-5, 1000)
    }

    return {"svc_1" : svc_1}

def process_value(val):

    if isinstance(val, (np.floating, float)):
        return np.round(val, 6)

    return val

def search_grid(n_parameters_by_model=15):

    grid = {}
    _grid = classical_grid()

    for model in _grid:

        param_list = list(
            ParameterSampler(
                _grid[model], n_iter=n_parameters_by_model, random_state=seed
            )
        )

        grid[model] = [
            dict(
                (k, [process_value(v)]) for (k, v) in d.items()
            ) for d in param_list
        ]

    return grid
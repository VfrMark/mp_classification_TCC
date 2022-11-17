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
from sklearn.mixture import GaussianMixture

#Config file
import config

#Fix seed to reproducibility
SEED = config._seed()

def get_grid():

    #Suppport vector machine with linear kernel
    svc = {
        'estimator': [
            SVC(
                kernel = 'linear'
                , probability=True
                , random_state = SEED
            )
        ]
        , 'estimator__C' : loguniform(1e-5, 1000)
    }

    return [svc]

def process_value(val):

    if isinstance(val, (np.floating, float)):
        return np.round(val, 6)

    return val

def search_grid(n_parameters_by_model=15):

    grid = {}
    _grid = get_grid()

    for model in _grid:

        param_list = list(
            ParameterSampler(
                model, n_iter=n_parameters_by_model, random_state=SEED
            )
        )

        grid[model] = [
            dict(
                (k, [process_value(v)]) for (k, v) in d.items()
            ) for d in param_list
        ]

    return grid
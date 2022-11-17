# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 08:50:53 2021

@author: Edson cilos
"""

#Standard Packages 
import numpy as np
from scipy.stats import randint
from scipy.stats import uniform
from scipy.stats import loguniform

#Sklearn API

from sklearn.svm import SVC
from sklearn.model_selection import ParameterSampler

#Tensorflow API
from tensorflow import keras
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.callbacks import EarlyStopping

#Config file
import config

#Fix seed to reproducibility
seed = config._seed()
rng = config._rng()

#Be aware about Keras's issue https://github.com/keras-team/keras/issues/13586
#Solution here https://stackoverflow.com/questions/62801440/kerasregressor-cannot-clone-object-no-idea-why-this-error-is-being-thrown/66771774#66771774




#Basic setup to build neural network
def build_nn(n_hidden = 1, 
             n_neurons = 50, 
             momentum = 0.9,
             learning_rate = 0.001, 
             act = "sigmoid"):
    
    model = keras.models.Sequential()
    
    for layer in range(int(n_hidden)):
        model.add(keras.layers.Dense(n_neurons, activation= act))
        
    model.add(keras.layers.Dense(14, activation="softmax"))
    
    optimizer = keras.optimizers.SGD(momentum = momentum,
                                     nesterov = True, 
                                     lr=learning_rate)
    
    model.compile(loss="sparse_categorical_crossentropy", 
                  optimizer=optimizer,
                  metrics=["accuracy"])
    
    return model


def neural_grid(epochs = 1000, patience = 3):

    #Create dictionary with base model, including parameter grid
    neural_network = {'estimator': [KerasClassifier(build_nn, 
                                 epochs = epochs,
                                 callbacks = [EarlyStopping(monitor='loss', 
                                                            patience= patience,
                                                            min_delta=0.001
                                                            )]
                                 )],
        "estimator__n_hidden": [1, 2, 3, 4, 5],
        "estimator__n_neurons": [10, 50, 100, 150, 200],
        "estimator__momentum" : np.arange(0.1, 0.9, 0.3),
        "estimator__learning_rate": [1e-3, 1e-2, 1e-1],
        "estimator__act": ["relu", "sigmoid", "tanh"],
        }
    
    return [neural_network]

#Basic grid structure for classical algorithms, expecpt neural network
def classical_grid():
  
    #Parameter C, used in several models
    c_list = loguniform(1e-5, 1000)

    #Suppport vector machine with linear kernel
    svc_1 = {
        'estimator': [
            SVC(
                kernel = 'linear'
                , probability=True
                , random_state = seed
            )
        ]
        , 'estimator__C' : c_list.copy() 
    }

    return [svc_1]

def process_value(val):

    if isinstance(val, (np.floating, float)):
        return np.round(val, 6)

    return val

def search_grid(n_inputs, n_parameters_by_model=15):

    grid = {}
    _grid = classical_grid(n_inputs)

    for model in _grid:

        param_list = list(
            ParameterSampler(
                _grid[model], n_iter=n_parameters_by_model, random_state=rng
            )
        )

        grid[model] = [
            dict(
                (k, [process_value(v)]) for (k, v) in d.items()
            ) for d in param_list
        ]

    return grid
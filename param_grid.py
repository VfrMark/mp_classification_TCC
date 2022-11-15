# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 08:50:53 2021

@author: Edson cilos
"""

#Standard Packages 
import numpy as np

#Sklearn API

from sklearn.svm import SVC

#Tensorflow API
from tensorflow import keras
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.callbacks import EarlyStopping

#Config file
import config

#Fix seed to reproducibility
seed = config._seed() 

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
    c_list = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

    #Suppport vector machine with linear kernel    
    svc_1 = {'estimator': [SVC(kernel = 'linear', probability=True, 
                               random_state = seed)],
             'estimator__C' : c_list.copy() 
             }
    
    #Suppport vector machine with non-linear kernel
    svc_2 = {'estimator': [SVC(probability=True, random_state = seed)],
             'estimator__C' : c_list.copy(),
             'estimator__kernel' : ['rbf', 'poly', 'sigmoid'],
             'estimator__gamma' : ['scale', 'auto']
        }

    return [svc_1, svc_2]

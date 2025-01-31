# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 21:37:51 2021

@author: Edson Cilos
update
"""
import os
import numpy as np

path_dic = {
    'mccv': os.path.join('results', 'mccv'),
    'grid_search': os.path.join('results', 'grid_search'),
    'graphics' : os.path.join('results', 'graphics')
    }

def _seed():
    return 42

def _rng():
    return np.random.RandomState(_seed)

def _scaler_list():
    return ['std']

def _get_path(name):
    
    folder_path = path_dic[name]

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        
    return folder_path
    
def _blue():
    return (1/255,209/255,209/255)

def _purple():
    return (90/255, 53/255, 182/255)
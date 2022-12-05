# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 14:09:51 2021

@author: Edson Cilos
"""
#Sklearn modules
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier

#Imblearn modules
from imblearn.pipeline import Pipeline

from gmm_oversampling import oversample_gmm

#Config file
import config

#Fix seed to reproducibility
SEED = config._seed()
        
    
def build_pipe():
    
    prefix = ''
    
    pre_pipe =[
        ('std', StandardScaler())
        , ('gmm', oversample_gmm())
        , ('estimator', DummyClassifier())
        ]

    return  Pipeline(pre_pipe), prefix

def pipe_config(file_name):
    
    values = file_name.split('_')

    over_sample = True if 'over' in values else False
    gmm = True if 'gmm' in values else False

    return over_sample, 
  
    
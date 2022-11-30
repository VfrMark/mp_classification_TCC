# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 14:09:51 2021

@author: Edson Cilos
"""
#Sklearn modules
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier

#Imblearn modules
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline

#Config file
import config

#Fix seed to reproducibility
seed = config._seed()
        
    
def build_pipe(over_sample = False, gmm = True):
    
    prefix = ''
    
    pre_pipe =[
        ('std', StandardScaler())
        , ('estimator', DummyClassifier())
        ]     

    if(over_sample):
        
       prefix += 'over_'

       pre_pipe.insert(-1, ('over_sample',
                            RandomOverSampler(random_state = seed)) 
                             )

    if(gmm):
        
       prefix += 'gmm_'

       pre_pipe.insert(-1, ('gmm_',
                            GaussianMixture(random_state = seed)) 
                             )

    return  Pipeline(pre_pipe), prefix

def pipe_config(file_name):
    
    values = file_name.split('_')

    over_sample = True if 'over' in values else False
    gmm = True if 'gmm' in values else False

    return over_sample, 
  
    
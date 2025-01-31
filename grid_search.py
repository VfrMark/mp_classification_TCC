#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 17:52:03 2020

@author: Edson Cilos
"""
#Standard modules
import os
import pandas as pd
from timeit import default_timer as timer

#Sklearn Model Selection
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold

#Project modules
from utils import file_name as f_name
from utils import append_time
from param_grid import search_grid
from pipeline import build_pipe

#Config module
import config

seed = config._seed()
gs_folder = config._get_path('grid_search')

def search(over_sample = True, 
           param_grid = search_grid(),
           prefix = '',
           n_jobs = 1,
           save = True
        ):
    
    
    print('Loading training set...')
    X_train = pd.read_csv(os.path.join('data', 'X_train.csv')) 
    y_train = pd.read_csv(os.path.join('data', 'y_train.csv')).values.ravel() 
    
    print('Building pipeline...')
    pipe, file_name = build_pipe(over_sample = over_sample)
    
    file_name = prefix + file_name + 'gs.csv'
    
    print('The file name is: ' + file_name)
      
    cv_fixed_seed = StratifiedKFold(n_splits = 5, shuffle = False)

    print('Running parameter search (It can take a long time) ...')
    search = RandomizedSearchCV(pipe,
                          param_grid,
                          scoring = 'neg_log_loss',
                          cv = cv_fixed_seed,
                          n_jobs = n_jobs,
                          verbose = 100,
                          random_state = seed
                        )

    search = search.fit(X_train, y_train)

    results = pd.concat([pd.DataFrame(search.cv_results_["params"]),
                     pd.DataFrame(search.cv_results_['std_test_score'], 
                                  columns=["std"]),
                     pd.DataFrame(search.cv_results_["mean_test_score"], 
                                  columns=["neg_log_loss"])],axis=1)
                     
    results.sort_values(by=['neg_log_loss'], ascending=False, inplace=True)
    
    if save:
        final_path = os.path.join(gs_folder, file_name)        
        print('Search is finished, saving results in ' + final_path)
        results.to_csv(final_path, index = False)
    
    return results

def run_gs():
    
    i = 0
    print('RadomizedSearch across several combinations')
          
    for over in [False, True]:
        
        i += 1
        
        grid = search_grid() # neural_grid() if nn else classical_grid() (Maybe use nn in future)
        
        file_name = f_name(over_sample= over)
    
        file_path = os.path.join(gs_folder, file_name)
        
        if os.path.isfile(file_path):
            print(file_name + " already exists, iteration was skipped ...")
            
        else:
            print("{0} iteration ({1}/32)...".format(file_name, str(i)))
            start = timer()
            search(over_sample = over, 
                   param_grid = grid['svc'],
                   n_jobs = 1)
            end = timer()
            append_time(file_name, str(end - start))
            
    print("RadomizedSearch fully finished...")

if __name__ == "__main__":
    run_gs()
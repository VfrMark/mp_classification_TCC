# sklearn tools needed
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin

# Standard libaries
import numpy as np
import pandas as pd
import config

SEED = config._seed()

class oversample_gmm(BaseEstimator, TransformerMixin):
    def __init__(self, n_class_var = 14) -> None:
        self.n_class = n_class_var

    def sample_numbers(self, X, y=None):
        dict_class_n = {}
        classe = []
        numbers = []

        classes = y['label'].unique()
    
        for sample_class in classes:
            n_class = (y['label'] == sample_class).sum()
            classe.append(sample_class)
            numbers.append(n_class)

        dict_class_n['class'] = classe
        dict_class_n['Quantity'] = numbers
    
        self.label_count = pd.DataFrame.from_dict(data=dict_class_n)
        self.label_count['max'] = self.label_count.max()['Quantity']
        self.label_count['n_samples_needed'] = self.label_count['max'] - self.label_count['Quantity']
        classes_to_sample = self.label_count[self.label_count['n_samples_needed'] != 0]
        class_most_n = self.label_count[self.label_count['n_samples_needed'] == 0]
        classes = classes_to_sample['class'].unique()

        return classes, classes_to_sample, class_most_n
    
    def fit(self, X, y=None, n_components=1):
        classes, _, _ = self.sample_numbers(X, y)

        self.gmm_models = {}

        for sample_class in classes:
            target_class_indices = np.flatnonzero(y == sample_class)
            
            X_filtered = X.iloc[target_class_indices]

            gmm = GaussianMixture(n_components
                , max_iter=10000
                , random_state=SEED
            )

            gmm.fit(X_filtered)
            
            self.gmm_models['gmm_' + str(sample_class)] = gmm

        return self.gmm_models

    
    def transform(self, X, y=None):
        #Implement transformation
        classes, samples_values, class_most_n = self.sample_numbers(X, y)

        X_resampled = X.copy()
        y_resampled = y.copy()
        for sample_class in classes:
            sample_numbers = samples_values[samples_values['class'] == sample_class]['n_samples_needed'].values[0]

            gmm_model = self.gmm_models['gmm_' + str(sample_class)]
            
            X_sampled_data = gmm_model.sample(sample_numbers)
            X_sampled_data = pd.DataFrame(X_sampled_data[0], columns = X_resampled.columns)
            
            preparing_labels = [sample_class for n in range(sample_numbers)]
            y_sampled_data = pd.DataFrame(preparing_labels, columns = ['label'])
        
            X_resampled = pd.concat([X_resampled, X_sampled_data], ignore_index=True)
            y_resampled = pd.concat([y_resampled, y_sampled_data], ignore_index=True)

        return X_resampled, y_resampled

import os
    
X_train = pd.read_csv(os.path.join('data', 'X_train.csv')) 
y_train = pd.read_csv(os.path.join('data', 'y_train.csv'))
gmm = oversample_gmm()

gmm.fit(X_train, y_train)
gmm.transform(X_train, y_train)
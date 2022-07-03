# -*- coding: utf-8 -*-
"""
Created on Sun May  1 17:58:19 2022

@author: marce
"""
import numpy as np
from PythonFiles.explainability_abstract_class import explainability

class omega_squared(explainability):
    
    def fit(self):
        # from first layer to output
        self.getInput_activation()
        
        layers = len(self.model.layers)
        
        # for the last hidden layer
        R_j = self.dic[f"{layers-1}_layer"]
        
        for layer in np.arange(layers-2,-1,-1):
            
            R_i = self.omegaRule(R_j, self.model.layers[layer].weights[0].numpy())
            R_j = R_i
            
        return R_i
    
    def omegaRule(self, R_j, weights):
        w_squared = weights*weights
        n = w_squared/np.sum(w_squared, axis = 0)
        
        return np.matmul(n, R_j)
        
    
    



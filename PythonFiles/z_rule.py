# -*- coding: utf-8 -*-
"""
Created on Sat May  7 13:10:48 2022

@author: Marcel Pommer
"""

import numpy as np
from PythonFiles.explainability_abstract_class import explainability
from tensorflow.keras import backend as K

class z_rule(explainability):
        
    def fit(self):
        self.getInput_activation()
        layers = len(self.model.layers)
        
        # for the last hidden layer
        R_j = self.dic[f"{layers-1}_layer"]
        
        for layer in np.arange(layers-2,-1,-1):
            X_i = self.dic[f"{layer}_layer"]
            
            R_i = self.z_Rule(R_j, self.model.layers[layer].weights[0].numpy(), X_i)
            R_j = R_i
            
        return R_i
    
    
    def z_Rule(self, R_j, weights,layer_output):
        
        z =  np.matmul(weights.T,layer_output)
        
        weights_times_relevenace_div_z = np.matmul(weights,R_j/z)
        
        times_layer_output = layer_output*weights_times_relevenace_div_z
                    
        return times_layer_output
    
        

# -*- coding: utf-8 -*-
"""
Created on Sun May  1 17:58:19 2022

@author: marce
"""
import numpy as np
from PythonFiles.explainability_abstract_class import explainability

class omega_squared(explainability):
    
    def fit(self):
        number_layers = len(self.model.layers)
        
        
        output = self.result
        for layer in np.arange(number_layers-1, -1, -1):
            output = self.omegaRule(output, self.model.layers[layer].weights[0].numpy())
            
        return output
    
    def omegaRule(self, R_j, weights):
        input_relevance = []
        
        summe = sum(weights**2)
        for in_neuron_index, inputs in enumerate(weights):
            R_i = 0
            for out_neuron_index, weight in enumerate(inputs):
                weight_squared = weight**2
                weight_sum = summe[out_neuron_index]
                R_i += weight_squared/weight_sum*R_j[out_neuron_index]
            
            
            input_relevance.append(R_i)
            
        return input_relevance
        




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
        R_j = self.dic["1_layer"]
        
        R_i = self.omegaRule(R_j, self.model.layers[0].weights[0].numpy())
            
        return R_i
    
    def omegaRule(self, R_j, weights):
        w_squared = weights*weights
        n = w_squared/np.sum(w_squared, axis = 0)
        
        return np.matmul(n, R_j)
        
        # summe = sum(weights**2)
        # for in_neuron_index, inputs in enumerate(weights):
        #     R_i = 0
        #     for out_neuron_index, weight in enumerate(inputs):
        #         weight_squared = weight**2
        #         weight_sum = summe[out_neuron_index]
        #         R_i += weight_squared/weight_sum*R_j[out_neuron_index]
            
            
        #     input_relevance.append(R_i)
            
        # return input_relevance
    
    def getInput_activation(self):
        dic = {}
        dic["0_layer"]= self.input_vector
        
        input_array = self.input_vector
        for layer in np.arange(1,len(self.model.layers)+1):
            weights = self.model.layers[layer-1].weights[0].numpy()
            bias = self.model.layers[layer-1].weights[1].numpy()
            
            output = np.matmul(input_array,weights)+bias
            output = np.array([max(0,x) for x in output])
            dic[f"{layer}_layer"]=output
            
            input_array = output
            
            
        self.dic = dic
    
    
    # def fit(self):
    #     number_layers = len(self.model.layers)
        
        
    #     output = self.result
    #     for layer in np.arange(number_layers-1, -1, -1):
    #         output = self.omegaRule(output, self.model.layers[layer].weights[0].numpy())
            
    #     return output
    
    # def omegaRule(self, R_j, weights):
    #     input_relevance = []
        
    #     summe = sum(weights**2)
    #     for in_neuron_index, inputs in enumerate(weights):
    #         R_i = 0
    #         for out_neuron_index, weight in enumerate(inputs):
    #             weight_squared = weight**2
    #             weight_sum = summe[out_neuron_index]
    #             R_i += weight_squared/weight_sum*R_j[out_neuron_index]
            
            
    #         input_relevance.append(R_i)
            
    #     return input_relevance




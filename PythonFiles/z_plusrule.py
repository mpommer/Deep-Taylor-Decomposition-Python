# -*- coding: utf-8 -*-
"""
Created on Sat Jun 11 17:49:28 2022

@author: marce
"""

# -*- coding: utf-8 -*-
"""
Created on Sat May  7 13:10:48 2022

@author: Marcel Pommer
"""

import numpy as np
from PythonFiles.explainability_abstract_class import explainability
from tensorflow.keras import backend as K

class z_plusrule(explainability):
    
    def __init__(self, model, result, input_vector):
        self.model = model
        self.result = result
        self.input_vector = input_vector
        
    def fit(self):
        self.getInput_activation()
        number_layers = len(self.model.layers)
        
       
        output = self.result
        for layer in np.arange(number_layers-1, -1, -1):
            layer_output = self.dic[f"{layer}_layer"]
            output = self.z_Rule(output, self.model.layers[layer].weights[0].numpy(), layer_output)
            
        return output
    
   
    # def z_Rule(self, R_j, weights,layer_output):
    #     try:
    #         R_j = [x[0] for x in R_j]
    #     except:
    #         pass
        
    #     for index in range(len(weights)):
    #         weights[index]*=layer_output[index]
            
    #     sums = weights.sum(axis=1)

    #     for index in range(len(weights)):
    #         weights[index]/=sums[index]  
    #         weights[np.isnan(weights)] = 0
    #         weights[index]*=R_j
        
    #     output = weights.sum(axis=1)
        
    #     return output
    
    def z_Rule(self, R_j, weights,layer_output):
        
        weights[weights<0] = 0
        z =  np.matmul(weights.T,layer_output)
        
        weights_times_relevenace_div_z = np.matmul(weights,R_j/z)
        
        times_layer_output = [x*y for (x,y) in zip(layer_output, weights_times_relevenace_div_z)]
                    
        return times_layer_output
    
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
        

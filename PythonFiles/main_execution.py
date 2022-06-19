# -*- coding: utf-8 -*-
"""
Created on Sun May  1 17:50:10 2022

@author: Marcel
"""
 
from PythonFiles.dataReading import data_reading_transformation
from PythonFiles.omega_squared import omega_squared
from PythonFiles.z_rule import z_rule 
from PythonFiles.z_plusrule import z_plusrule 
import pandas as pd
from PythonFiles.Neural_Network import NN
import numpy as np

path_url = "data/titanic"

minmaxscaler=True

x_train, y_train, x_test, y_test, feature_name = data_reading_transformation(path_url, minmaxscaler, withName = True)

neural_net = NN(x_train, y_train)

neural_net.fit(layers = 1, neurons = [10]*1)

model = neural_net.getModel()

neural_net.testResult(x_test,y_test)

prediction = neural_net.predict(x_test)

w1 = model.layers[0].weights[0].numpy()
b1 = model.layers[0].weights[1].numpy()
w2 = model.layers[1].weights[0].numpy()
b2 = model.layers[1].weights[1].numpy()

#%% omega ^2- rule

passenger = 89
taylor_dec = omega_squared(model, prediction[passenger], x_test[passenger])
        
res = taylor_dec.fit()
        
        
taylor_dec.printExplainability(feature_name)






#%%


for i in range(5):
    if  prediction[i]>0:
        taylor_dec = z_rule(model, prediction[i],x_test[i])
        
        res = taylor_dec.fit()
        print(res)
        
        taylor_dec.printExplainability(feature_name)

#%%

taylor_dec = z_rule(model, prediction[14],x_test[14])
        
res = taylor_dec.fit()

taylor_dec.printExplainability(feature_name)

#%%
taylor_dec = z_plusrule(model, prediction[14],x_test[14])
        
res = taylor_dec.fit()

taylor_dec.printExplainability(feature_name)


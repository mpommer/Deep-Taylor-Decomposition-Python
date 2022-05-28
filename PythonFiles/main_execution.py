# -*- coding: utf-8 -*-
"""
Created on Sun May  1 17:50:10 2022

@author: Marcel
"""
 
from PythonFiles.dataReading import data_reading_transformation
from PythonFiles.omega_squared import omega_squared
from PythonFiles.Neural_Network import NN
import numpy as np

path = "data/titanic"

minmaxscaler=True

x_train, y_train, x_test, y_test, feature_name = data_reading_transformation(path, minmaxscaler)

neural_net = NN(x_train, y_train)

neural_net.fit(layers = 1, neurons = [5]*5)

model = neural_net.getModel()

neural_net.testResult(x_test,y_test)

prediction = neural_net.predict(x_test)

#%%

for i in range(1):
    if  prediction[i]>0:
        taylor_dec = omega_squared(model, prediction[i])
        
        res = taylor_dec.fit()
        
        
        taylor_dec.printExplainability(feature_name)






#%%

from z_rule import z_rule

for i in range(10):
    if  prediction[i]>0:
        taylor_dec = z_rule(model, prediction[i],x_test[i])
        
        res = taylor_dec.fit()
        print(res)
        
        # taylor_dec.printExplainability(feature_name)


taylor_dec = z_rule(model, prediction[14],x_test[14])
        
res = taylor_dec.fit()




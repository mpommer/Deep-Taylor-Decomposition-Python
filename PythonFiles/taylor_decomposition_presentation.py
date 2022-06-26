# -*- coding: utf-8 -*-
"""
Created on Sun May  1 17:50:10 2022

@author: Marcel Pommer
"""

from PythonFiles.dataReading import data_reading_transformation
from PythonFiles.omega_squared import omega_squared
from PythonFiles.Neural_Network import NN

# preprocess data
x_train, y_train, x_test, y_test, feature_name = data_reading_transformation(path_url = "data/titanic", minmaxscaler = True, withID = True)

# build the neural network
neural_network = NN(x_train, y_train)
neural_network.fit(layers = 1, neurons = [5]*1, verbose = 0)

# print test results
neural_network.testResult(x_test,y_test)
neural_network.trainResult()

# get the model
model = neural_network.getModel()

# get predictions
prediction = neural_network.predict(x_test)


#%% omega ^2- rule
# choose passenger and create taylor decomposition
passenger = 89
taylor_decomposition = omega_squared(model, prediction[passenger], x_test[passenger])
        
relevance = taylor_decomposition.fit()
        
        
taylor_decomposition.printExplainability(feature_name)




# -*- coding: utf-8 -*-
"""
Created on Thu May 26 17:20:17 2022

@author: Marcel Pommer
"""

import innvestigate
from PythonFiles.dataReading import data_reading_transformation
from PythonFiles.omega_squared import omega_squared
from PythonFiles.Neural_Network import NN
import numpy as np
import tensorflow as tf


path = "data/titanic"

minmaxscaler=True

x_train, y_train, x_test, y_test, feature_name = data_reading_transformation(path, minmaxscaler)

neural_net = NN(x_train, y_train)

neural_net.fit(layers = 1, neurons = [10])

model = neural_net.getModel()

neural_net.testResult(x_test,y_test)

prediction = neural_net.predict(x_test)



#%% 

analyzer = innvestigate.create_analyzer("deep_taylor", model)

analysis = analyzer.analyze(x_test[14])



#%%

import imp
import numpy as np
import os

import keras
import keras.backend
import keras.models

import innvestigate
import innvestigate.utils as iutils

# Use utility libraries to focus on relevant iNNvestigate routines.
eutils = imp.load_source("utils", "../utils.py")
mnistutils = imp.load_source("utils_mnist", "../utils_mnist.py")











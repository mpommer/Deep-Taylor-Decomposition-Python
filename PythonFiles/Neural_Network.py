# -*- coding: utf-8 -*-
"""
Created on Sun May  1 17:57:47 2022

@author: Marcel Pommer
"""
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense
from tensorflow import keras
import tensorflow as tf
import pandas as pd
seed = 1864
tf.random.set_seed(seed)
import numpy as np
np.random.seed(seed)

class NN:
    
    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        
    def fit(self, layers, neurons):

        model = keras.Sequential()
        for l in range(layers):
            model.add(Dense(neurons[l], input_dim=self.x_train.shape[1], activation = "ReLU"))
            
        model.add(Dense(1, activation='sigmoid', name='output'))

        model.compile(loss="mean_squared_error", optimizer= "adam", metrics=['mean_absolute_error','mean_squared_error'])

        es = EarlyStopping(monitor='val_loss', mode='min',min_delta=1e-12, patience=100, restore_best_weights=True)
                            
        history = model.fit(self.x_train, self.y_train, epochs=500, batch_size=1, verbose=0, validation_split=0.2, callbacks=[es])

        mae = pd.DataFrame(history.history).copy()
        
        self.model = model

    def getModel(self):
        return self.model
    
    def predict(self, x):
        prediction = self.model.predict(x)
        
        return prediction      
        
        
    def trainResult(self):
        train_result = self.model.predict(self.x_train).round()

        train = [1 if x ==y else 0 for (x,y) in zip(train_result,self.y_train)]
        
        return print(sum(train)/len(train))

    def testResult(self, x_test, y_test):
        test_result = self.model.predict(x_test).round()

        test = [1 if x ==y else 0 for (x,y) in zip(test_result,y_test.iloc[:,1])]
        
        return print(sum(test)/len(test))


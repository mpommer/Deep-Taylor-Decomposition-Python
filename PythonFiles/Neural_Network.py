# -*- coding: utf-8 -*-
"""
Created on Sun May  1 17:57:47 2022

@author: Marcel Pommer
"""
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense
from tensorflow import keras
from tensorflow.keras import initializers
import tensorflow as tf
import pandas as pd
seed = 12345


class NonPos(tf.keras.constraints.Constraint):
  """Constrains weight tensors to be centered around `ref_value`."""

  def __init__(self):
      pass
  def __call__(self, w):
      
     w = tf.cast(w < 0, w.dtype) * w
     return w
 

  def get_config(self):
    return {'ref_value': self.ref_value}


class NN:
    
    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        
    def fit(self, layers, neurons, verbose = 0):
        tf.random.set_seed(seed)

        model = keras.Sequential()
        model.add(Dense(neurons[0], input_dim=self.x_train.shape[1], activation = "ReLU", bias_constraint=NonPos()))
        if layers>1:
            for layers in range(layers-1):
                model.add(Dense(neurons[layers+1], input_dim=neurons[layers], activation = "ReLU", bias_constraint=NonPos()))


        # for l in range(layers):
        #     model.add(Dense(neurons[l], input_dim=self.x_train.shape[1], activation = "ReLU", bias_constraint=NonNeg()))
            
        # model.add(Dense(1, input_dim=neurons[-1],activation="ReLU", trainable=False, name='output'))
        model.add(Dense(1, input_dim=neurons[-1],
                        kernel_initializer=initializers.RandomNormal(mean=1., stddev=0.),
                        bias_initializer=initializers.Zeros(),
                        activation="linear", 
                        trainable=False, 
                        name='output'))

        # model.add(AveragePooling1D(pool_size=2, strides=1, name = "output"))

        model.compile(loss="mean_squared_error", optimizer= "adam", metrics=['mean_squared_error'])

        es = EarlyStopping(monitor='mean_squared_error', mode='min',min_delta=1e-8, patience=25, restore_best_weights=True)
                            
        history = model.fit(self.x_train, self.y_train, epochs=250, batch_size=1, verbose=verbose, validation_split=0.2, callbacks=[es])

        mae = pd.DataFrame(history.history).copy()
        
        self.model = model

    def getModel(self):
        return self.model
    
    def predict(self, x):
        prediction = self.model.predict(x)
        
        return prediction      
        
        
    def trainResult(self):
        train_result = [1 if x >0.5 else 0 for x in self.model.predict(self.x_train)]

        train = [1 if x ==y else 0 for (x,y) in zip(train_result,self.y_train)]
        
        return print(sum(train)/len(train))

    def testResult(self, x_test, y_test):
        test_result = [1 if x >0.5 else 0 for x in self.model.predict(x_test)]

        test = [1 if x ==y else 0 for (x,y) in zip(test_result,y_test.iloc[:,1])]
        
        return print(sum(test)/len(test))





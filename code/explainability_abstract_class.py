# -*- coding: utf-8 -*-
"""
Created on Sat May  7 12:52:03 2022

@author: Marcel Pommer
"""

import abc 
from abc import abstractmethod
from tabulate import tabulate

class explainability(metaclass=abc.ABCMeta):
    
    def __init__(self, model, result):
        self.model = model
        self.result = result
        
        
    @abstractmethod
    def fit(self):
        pass
    
    
    def returnModel(self):
        return self.model
    
    
    def returnResult(self):
        return self.result
        
            
    def printExplainability(self, feature_names):
        
        table = [["feature", "absolute relevance", "relevance in perc"]]
        
        relevance = self.fit()
        result = self.returnResult()
        for index, name in enumerate(feature_names):
            value = [name, round(relevance[index],4), f"{round(relevance[index]/result[0], 4)*100} %"]
            table.append(value)
            
        print(tabulate(table))
            
    def isConservative(self):
        relevance = self.fit()
        
        bol = abs(sum(relevance)-self.result)<0.01
        if bol:
            print("The model is conservative with a threshold of 1 %")
        else:
            print("The model is not conservative with a threshold of 1 %")
        
    def isPositive(self):
        relevance = self.fit()
        
        bol = (relevance >=-0.001).all()
        if bol:
            print("The model is positive with a threshold of 0.1 %")
        else:
            print("The model is not positive with a threshold of 0.1 %")
    
    def isConsistent(self):
        relevance = self.fit()
        
        positive = (relevance >=-0.001).all()
        conservative = abs(sum(relevance)-self.result)<0.01
        if positive and conservative:
            print("The model is consistent!")
        else:
            print("The model is not consistent!")
    
    
    
    
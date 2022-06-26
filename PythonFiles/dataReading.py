# -*- coding: utf-8 -*-
"""
Created on Sun May  1 12:10:49 2022

@author: Marcel Pommer
"""

from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
pd.set_option('display.max_rows', 15)
pd.set_option('display.max_columns', 15)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', lambda x: "%.2f" % x)


def data_reading_transformation(path_url, minmaxscaler=False, withID=False):
    df_gender_submission = pd.read_csv(path_url + "/gender_submission.csv")
    y_test = df_gender_submission
    df_train = pd.read_csv(path_url + "/train.csv")
    df_test = pd.read_csv(path_url + "/test.csv")
    
    if withID:
        df_train_reduced = df_train.drop(labels=["Name", "Cabin", "Ticket"], axis=1)
        df_test_reduced = df_test.drop(labels=["Name", "Cabin", "Ticket"], axis=1)
    else:        
        df_train_reduced = df_train.drop(labels=['Name', "Ticket", "Cabin", "PassengerId"], axis=1)
        df_test_reduced = df_test.drop(labels=['Name', "Ticket", "Cabin", "PassengerId"], axis=1)
        
    
    ageMeanDad = int(df_train_reduced[(df_train_reduced["Survived"]==0)][[ "Age"]].mean())
    ageMeanSurvived = int(df_train_reduced[(df_train_reduced["Survived"]==1)][[ "Age"]].mean())
       
    #!!! not performant 
    df_train_reduced[["Age"]] = df_train_reduced[["Age"]].replace(np.NaN, -1)
    df_train_reduced.loc[(df_train_reduced.Survived ==0) &(df_train_reduced.Age == -1), 'Age'] = ageMeanDad
    df_train_reduced.loc[(df_train_reduced.Survived ==1) &(df_train_reduced.Age == -1), 'Age'] = ageMeanSurvived
    
    
    meanAge = int(df_test_reduced.loc[(df_test_reduced.Age >0), 'Age'].mean())
    df_test_reduced[["Age"]] = df_test_reduced[["Age"]].replace(np.NaN, meanAge)
    
    
    meanFare = df_test_reduced.loc[(df_test_reduced.Fare >0), 'Fare'].mean()
    df_test_reduced[["Fare"]] = df_test_reduced[["Fare"]].replace(np.NaN, meanFare)
   
    # most embarked at S, replace nan with S (southhampton)
    df_train_reduced[["Embarked"]] = df_train_reduced[["Embarked"]].replace(np.NaN, "S")
    
    
    # creating instance of one-hot-encoder
    enc = OneHotEncoder(handle_unknown='ignore')
    
    enc_df_train = pd.DataFrame(enc.fit_transform(df_train_reduced[['Embarked']]).toarray())
    
    enc_df_test = pd.DataFrame(enc.fit_transform(df_test_reduced[['Embarked']]).toarray())
    
    
    df_train_reduced[["Cherbourg","Queenstown", "Southampton"]] = enc_df_train
    df_test_reduced[["Cherbourg","Queenstown", "Southampton"]] = enc_df_test
    
    df_train_reduced = df_train_reduced.drop(labels=['Embarked'], axis=1)
    df_test_reduced = df_test_reduced.drop(labels=['Embarked'], axis=1)
    
    
    df_train_reduced.loc[(df_train_reduced.Sex == "female"), 'Sex'] = 0
    df_train_reduced.loc[(df_train_reduced.Sex == "male"), 'Sex'] = 1
    df_train_reduced[['Sex']] = df_train_reduced[['Sex']].astype(int)
    
    
    df_test_reduced.loc[(df_test_reduced.Sex == "female"), 'Sex'] = 0
    df_test_reduced.loc[(df_test_reduced.Sex == "male"), 'Sex'] = 1
    df_test_reduced[['Sex']] = df_test_reduced[['Sex']].astype(int)
    
    y_train = df_train_reduced[["Survived"]].values.ravel()

    if minmaxscaler:
        scaler = MinMaxScaler(feature_range=(0,1))
    else:
        scaler = StandardScaler()
    
    X = df_train_reduced.drop(labels = "Survived", axis = 1)
    
    feature_names = X.columns
    
    scaler.fit(X)

    x_train = scaler.transform(X)
    x_test = scaler.transform(df_test_reduced)

    return x_train, y_train, x_test, y_test, feature_names





# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 10:01:11 2022

@author: Marcel Pommer
"""

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
import sklearn
import seaborn as sns
pd.set_option('display.max_rows', 15)
pd.set_option('display.max_columns', 15)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', lambda x: "%.2f" % x)

df_gender_submission = pd.read_csv("data/Titanic/titanic/gender_submission.csv")
df_train = pd.read_csv("data/Titanic/titanic/train.csv")
df_test = pd.read_csv("data/Titanic/titanic/test.csv")

df_gender_submission.head()
df_train.head()
df_test.head()

df_train_reduced = df_train.drop(labels=['Name', "Ticket", "Cabin"], axis=1)
df_test_reduced = df_test.drop(labels=['Name', "Ticket", "Cabin"], axis=1)

df_train_reduced.describe()

df_train_reduced.isnull().sum()
df_test_reduced.isnull().sum()


ageMeanDad = int(df_train_reduced[(df_train_reduced["Survived"]==0)][[ "Age"]].mean())
ageMeanSurvived = int(df_train_reduced[(df_train_reduced["Survived"]==1)][[ "Age"]].mean())
df_train_reduced[["Age"]].mean()

df_train_reduced[["Age"]] = df_train_reduced[["Age"]].replace(np.NaN, -1)
df_train_reduced.loc[(df_train_reduced.Survived ==0) &(df_train_reduced.Age == -1), 'Age'] = ageMeanDad
df_train_reduced.loc[(df_train_reduced.Survived ==1) &(df_train_reduced.Age == -1), 'Age'] = ageMeanSurvived


meanAge = int(df_test_reduced.loc[(df_test_reduced.Age >0), 'Age'].mean())
df_test_reduced[["Age"]] = df_test_reduced[["Age"]].replace(np.NaN, meanAge)
# df_test_reduced.loc[(df_test_reduced.Age == -1), 'Age'] = meanAge

meanFare = df_test_reduced.loc[(df_test_reduced.Fare >0), 'Fare'].mean()
df_test_reduced[["Fare"]] = df_test_reduced[["Fare"]].replace(np.NaN, meanFare)
# df_test_reduced.loc[(df_test_reduced.Age == -1), 'Age'] = meanAge



df_train_reduced[["Embarked"]].value_counts()
# most embarked at S
df_train_reduced[["Embarked"]] = df_train_reduced[["Embarked"]].replace(np.NaN, "S")


# creating instance of one-hot-encoder
enc = OneHotEncoder(handle_unknown='ignore')

enc_df_train = pd.DataFrame(enc.fit_transform(df_train_reduced[['Embarked']]).toarray())

enc_df_test = pd.DataFrame(enc.fit_transform(df_test_reduced[['Embarked']]).toarray())


df_train_reduced[["C","Q", "S"]] = enc_df_train
df_test_reduced[["C","Q", "S"]] = enc_df_test

df_train_reduced = df_train_reduced.drop(labels=['Embarked'], axis=1)
df_test_reduced = df_test_reduced.drop(labels=['Embarked'], axis=1)


df_train_reduced.loc[(df_train_reduced.Sex == "female"), 'Sex'] = 0
df_train_reduced.loc[(df_train_reduced.Sex == "male"), 'Sex'] = 1
df_train_reduced[['Sex']] = df_train_reduced[['Sex']].astype(int)


df_test_reduced.loc[(df_test_reduced.Sex == "female"), 'Sex'] = 0
df_test_reduced.loc[(df_test_reduced.Sex == "male"), 'Sex'] = 1
df_test_reduced[['Sex']] = df_test_reduced[['Sex']].astype(int)

#%%
y = df_train_reduced[["Survived"]].values.ravel()
from sklearn.preprocessing import StandardScaler


scalar = StandardScaler()
# X = pd.DataFrame(scalar.fit_transform(df_train_reduced.drop(["Survived"],axis = 1),),
#         columns=df_train_reduced.columns.drop("Survived"))

X = df_train_reduced.drop(labels = "Survived", axis = 1)


from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 1/3, random_state=42)




models = []
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier(random_state = 12345)))
models.append(('RF', RandomForestClassifier(random_state = 12345)))
models.append(('SVM', SVC(gamma='auto', random_state = 12345)))
models.append(('XGB', GradientBoostingClassifier(random_state = 12345)))
models.append(("LightGBM", LGBMClassifier(random_state = 12345)))


result_train = []
result_test = []
names = []

for name, model in models:
    modelfit = model.fit(X_train, y_train)
    
    result_train.append(model.score(X_train, y_train))
    result_test.append(model.score(X_test, y_test))
    names.append(name)


param_grid = {"n_estimators" :[100,200,500,1000], 
             "max_features": [3,5,7], 
             "min_samples_split": [2,5,10,30],
            "max_depth": [3,5,8,None]}


gs_forest = GridSearchCV(RandomForestClassifier(random_state = 12345),
                         param_grid =param_grid,
                         n_jobs = -1,
                         cv = 10,
                         verbose = 0).fit(X, y)


rf_tuned = RandomForestClassifier(**gs_forest.best_params_)
fit_forest = rf_tuned.fit(X, y)


X_given_test = df_test_reduced
y_given_test = df_gender_submission[["Survived"]].values.ravel()

trainForest = fit_forest.score(X, y)
testForest = fit_forest.score(X_given_test, y_given_test)

print("Train precision: {} and test precision {}".format(trainForest, testForest))









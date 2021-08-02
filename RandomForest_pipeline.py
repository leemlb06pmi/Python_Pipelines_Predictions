#Code based on Templates from Machine Learning A-Z class

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing to a dataframe
dataset = pd.read_csv('Data.csv') #example, data not included in this script

#the final column is typically occupied by the output which we will be predicting
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#splitting the data for a supervised learning model
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#initializing and training a random forrest regression model
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10)
regressor.fit(X_train, y_train)

#predicting results and comparing 1-to-1 with the test set actual values
y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

#computing a r-score 
from sklearn.metrics import r2_score
r2_score(y_test, y_pred)

#%%
import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense

def format_concrete_data():
  '''formats data to prep for neural network processing'''
  concrete_data = pd.read_csv('https://ibm.box.com/shared/static/svl8tu7cmod6tizo6rk0ke4sbuhtpdfx.csv')
  concrete_data.head()
  concrete_data.shape
  concrete_data.describe()
  concrete_data.isnull().sum()
  concrete_data_columns = concrete_data.columns
  predictors = concrete_data[concrete_data_columns[concrete_data_columns != 'Strength']] # all columns except Strength
  target = concrete_data['Strength'] # Strength column
  predictors.head()
  target.head()
  predictors_norm = (predictors - predictors.mean()) / predictors.std()
  predictors_norm.head()
  n_cols = predictors_norm.shape[1] # number of predictors
  return predictors_norm, target, n_cols

def regression_model(n_cols):
    '''create and compile neural network model'''
    model = Sequential()
    model.add(Dense(50, activation='relu', input_shape=(n_cols,)))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model
    
predictors_norm, target, n_cols = format_concrete_data()
model = regression_model(n_cols)
model.fit(predictors_norm, target, validation_split=0.33, epochs=10, verbose=2)


# %%

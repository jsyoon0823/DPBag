"""DPBag Codebase.

Reference: James Jordon, Jinsung Yoon, Mihaela van der Schaar, 
"Differentially Private Bagging: Improved utility and cheaper privacy than subsample-and-aggregate", 
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8684-differentially-private-bagging-improved-utility-and-cheaper-privacy-than-subsample-and-aggregate

Last updated Date: April 26th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)

-----------------------------

data_loading.py

Note: Load and preprocess UCI adult data (Predict income > 50k)

(1) data_loading_adult: Load adult data.
  - Data reference: https://archive.ics.uci.edu/ml/datasets/adult
  - Code reference: https://yanhan.github.io/posts/2017-02-15-analysis-of-the-adult-data-set-from-uci-machine-learning-repository.ipynb.html
"""

# Necessary packages
import pandas as pd
import numpy as np


def data_loading_adult():
  """Load and preprocess UCI adult data (Predict income > 50k).
  
  Returns:
    - train_x: training features
    - train_y: training labels
    - valid_x: validation features
    - valid_y: validation labels
    - test_x: testing features
    - test_y: testing labels
  """
  
  ## Load raw datasets
  raw_data1 = pd.read_csv('data/adult.data', header = None, delimiter = ",")
  raw_data2 = pd.read_csv('data/adult.test', header = None, delimiter = ",")
  
  # Merge two datasets
  df = pd.concat((raw_data1, raw_data2), axis = 0)
  
  ## Define column names
  df.columns = ["Age", "WorkClass", "fnlwgt", "Education", "EducationNum",
                "MaritalStatus", "Occupation", "Relationship", "Race", "Gender",
                "CapitalGain", "CapitalLoss", "HoursPerWeek", "NativeCountry", "Income"]
  
  # Index reset
  df = df.reset_index()
  df = df.drop("index", axis = 1)
  
  ## Define label
  y = np.ones([len(df),])
  # Set >50K as 1 and <=50K as 0
  y[df["Income"].index[df["Income"] == " <=50K"]] = 0
  y[df["Income"].index[df["Income"] == " <=50K."]] = 0
    
  # Drop feature which can directly infer label
  df.drop("Income", axis=1, inplace=True,)
  
  ## Transform the type from string to float
  df.Age = df.Age.astype(float)
  df.fnlwgt = df.fnlwgt.astype(float)
  df.EducationNum = df.EducationNum.astype(float)
  df.HoursPerWeek = df.HoursPerWeek.astype(float)
  df.CapitalGain = df.CapitalGain.astype(float)
  df.CapitalLoss = df.CapitalLoss.astype(float)
  
  # One hot incoding for some categorical features
  df = pd.get_dummies(df, columns=["WorkClass", "Education", "MaritalStatus", "Occupation", "Relationship",
                                   "Race", "Gender", "NativeCountry"])
    
  # Treat feature as numpy array
  x = np.asarray(df)
        
  ## Normalization with Minmax Scaler
  for i in range(len(x[0,:])):
    x[:,i] = x[:,i] - np.min(x[:,i])
    x[:,i] = x[:,i] / (np.max(x[:,i]) + 1e-8)
    
  # Divide the data into train, valid, and test set (1/3 each)
  idx = np.random.permutation(len(y))
    
  # train
  train_x = x[idx[:int(len(y)/3)],:]
  train_y = y[idx[:int(len(y)/3)]]    
    
  # valid
  valid_x = x[idx[int(len(y)/3):(2*int(len(y)/3))],:]
  valid_y = y[idx[int(len(y)/3):(2*int(len(y)/3))]]
    
  # test
  test_x = x[idx[(2*int(len(y)/3)):],:]
  test_y = y[idx[(2*int(len(y)/3)):]]

  # Return train, valid, and test sets
  return train_x, train_y, valid_x, valid_y, test_x, test_y 
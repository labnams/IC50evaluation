#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
import random
from pandas import DataFrame
from datetime import datetime


workdir = "/home/corea/src/MDG_ML/dataset"


# In[2]:


dataset = np.load(workdir + "//MDG160k_190320_cls4_druginfo_change.npz")
ss0 = np.load(workdir + '//MDG160k_190315_shuffle_split_r0.npz')


# In[3]:


x = dataset['x']
y = dataset['y_lnIC50']
# y_linear = dataset['y_lnIC50']
ss0_train = ss0['train']
ss0_test = ss0['test']


# In[4]:


training_image_array, training_label_array = x[ss0_train], y[ss0_train]
test_image_array, test_label_array = x[ss0_test], y[ss0_test]

# # In[9]:
# ori = training_image_array
# bat = np.zeros((ori.shape[0],178))
# cat = np.hstack([ori,bat])
# training_image_array = cat

# # In[8]:
# training_image_array.shape

# # In[10]:
# ori2 = test_image_array
# bat2 = np.zeros((ori2.shape[0],178))
# cat2 = np.hstack([ori2,bat2])
# test_image_array = cat2


# In[5]:


# In[15]:
ab =[]
for i in range(50,100):
    ab.append(len(training_image_array) % i)
    
print(min(ab), ab.index(min(ab)))


# In[6]:


x.shape


# In[7]:


training_image_array.shape


# In[8]:


train_X, train_y, test_X, test_y = training_image_array, training_label_array, test_image_array, test_label_array

# if K.image_data_format() == 'channels_first':
#     train_X = train_X.reshape(train_X.shape[0], 1, img_rows, img_cols)
#     test_X = test_X.reshape(x_test.shape[0], 1, img_rows, img_cols)
#     input_shape = (1, img_rows, img_cols)
# else:
#     train_X = train_X.reshape(train_X.shape[0], img_rows, img_cols, 1)
#     test_X = test_X.reshape(test_X.shape[0], img_rows, img_cols, 1)
#     input_shape = (img_rows, img_cols, 1)


# In[10]:


# train_X = train_X.reshape(train_X.shape[0], train_X.shape[1], 1)
# test_X = test_X.reshape(test_X.shape[0], test_X.shape[1], 1)
# #input_shape = (img_rows, img_cols, 1)


# In[11]:


# training_label_array = training_label_array.reshape(training_label_array.shape[0], 1)
# test_label_array = test_label_array.reshape(test_label_array.shape[0], 1)
# # #input_shape = (img_rows, img_cols, 1)


# In[12]:


print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# The proportion of feature selection
perc = 50

# In[10]:


# Feature selection using f-regression
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import SelectKBest, SelectFdr, SelectPercentile, SelectFromModel

# selection the percentile feature

Starttime = datetime.now()
print(Starttime)

sel = SelectPercentile(f_regression, percentile=perc).fit(train_X, train_y)
Endtime = datetime.now()
print(Endtime)
train_X = sel.transform(train_X)
train_X.shape

test_X = sel.transform(test_X)
test_X.shape

print(train_X.shape, test_X.shape)



# In[13]:


train_X = train_X.astype('float32')
test_X = test_X.astype('float32')
# train_X /= 255
# test_X /= 255
print('train_X shape:', train_X.shape)
print(train_X.shape[0], 'train samples')
print(test_X.shape[0], 'test samples')


# hyperparameter tuning
n_esti = 250

StartTime8 = datetime.now()
print("StartTime :", StartTime8)
# Import the model we are using
from sklearn.ensemble import RandomForestRegressor
# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(random_state = 42
			   , n_jobs = 20
                           , n_estimators = n_esti)
# Train the model on training data
rf.fit(train_X, training_label_array)
EndTime8 = datetime.now()
print("EndTime :", EndTime8)


workdir = "/home/corea/src/MDG_ML/result/RF"


import pickle
from sklearn.externals import joblib
joblib.dump(rf, workdir + '//MDG160K_RF_best_model_fs_%s.pkl' % perc) 

# In[18]:


predicted_classes = rf.predict(test_X)


# In[19]:


predicted_value = predicted_classes


# In[20]:


a = pd.DataFrame(predicted_value)
b = pd.DataFrame(test_label_array)
c = pd.concat([a,b], axis=1)
c.columns=["Predicted","Test"]


# In[21]:


c.to_csv(workdir + '//MDG160K_RF_%s_result_fs_%.csv' % (n_esti, perc))


from scipy.stats import linregress
linregress(b[0], a[0])


# In[24]:


from sklearn.metrics import r2_score
r2_value = r2_score(b,a)
print(r2_value)



# In[26]:


rse = ((b[0]-a[0])**2).sum()
mse = rse / len(b)
print("Final rmse value is =",np.sqrt(mse))






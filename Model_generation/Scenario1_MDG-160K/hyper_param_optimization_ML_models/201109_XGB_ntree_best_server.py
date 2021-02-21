#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# In[ ]:


dataset.keys()


# In[ ]:


x = dataset['x']
y = dataset['y_lnIC50']
# y_linear = dataset['y_lnIC50']
ss0_train = ss0['train']
ss0_test = ss0['test']


# In[ ]:


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


# In[ ]:


# In[15]:
ab =[]
for i in range(100,200):
    ab.append(len(training_image_array) % i)
    
print(min(ab), ab.index(min(ab)))


# In[8]:


x.shape


# In[9]:


training_image_array.shape


# In[12]:


train_X, train_y, test_X, test_y = training_image_array, training_label_array, test_image_array, test_label_array

print(train_X.shape, test_X.shape)



train_X = train_X.astype('float32')
test_X = test_X.astype('float32')
# train_X /= 255
# test_X /= 255
print('train_X shape:', train_X.shape)
print(train_X.shape[0], 'train samples')
print(test_X.shape[0], 'test samples')


# In[15]:


StartTime8 = datetime.now()
print("StartTime :", StartTime8)

n_tree = 150
max_depth = 8

import xgboost as xgb
model = xgb.XGBRegressor(seed=42,
			nthread = 20,
			n_estimators = n_tree,
			max_depth = max_depth)
model.fit(train_X,training_label_array)
EndTime8 = datetime.now()
print("EndTime :", EndTime8)



test_eval = model.score(test_X,test_y)


# In[19]:



workdir = "/home/corea/src/MDG_ML/result/XGB"



# In[20]:


import matplotlib
from matplotlib import pyplot as plt


# In[21]:


predicted_classes = model.predict(test_X)


# In[22]:
import pickle
from sklearn.externals import joblib
joblib.dump(model, workdir + '//MDG160K_XGB_best_model.pkl') 



predicted_value = predicted_classes


# In[23]:


test_y


# In[24]:


a = pd.DataFrame(predicted_value)
b = pd.DataFrame(test_y)
c = pd.concat([a,b], axis=1)
c.columns=["Predicted","Test"]


# In[25]:


c.to_csv(workdir + '//MDG160K_XGB_nt_%s_md_%s_result.csv' % (n_tree, max_depth))

# In[26]:


c


# In[27]:


predicted_value.shape


# In[28]:


from scipy.stats import linregress
print(linregress(b[0], a[0]))


# In[29]:


from sklearn.metrics import r2_score
r2_value = r2_score(b,a)
print(r2_value)

rse = ((b[0]-a[0])**2).sum()
mse = rse / len(b)
print("RMSE: ",np.sqrt(mse))


# In[31]:




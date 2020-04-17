
# coding: utf-8

# In[1]:


# import dependencies
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import load_boston


# In[2]:


# understanding the datasets
boston = load_boston()
print(boston.DESCR)


# In[3]:


# access attributes of data
dataset = boston.data
for name, index in enumerate(boston.feature_names):
    print(index,name)


# In[5]:


# reshaping the data
data = dataset[:,9].reshape(-1,1)


# In[6]:


# shape of the data
np.shape(dataset)


# In[7]:


# target values
target = boston.target.reshape(-1,1)


# In[8]:


# shape of target
np.shape(target)


# In[9]:


# ensuring that matplot works inside notebook
get_ipython().run_line_magic('matplotlib', 'inline')
plt.scatter(data, target, color = 'blue')
plt.xlabel('full value property tax-rate')
plt.ylabel('Cost of house')
plt.show()


# In[10]:


# regression
from sklearn.linear_model import LinearRegression

# creating a regression model
reg = LinearRegression()

# fitting the model
reg.fit(data, target)


# In[11]:


# prediction 
pred = reg.predict(data)


# In[13]:


# ensuring the matplotlib is working well inside notebook
get_ipython().run_line_magic('matplotlib', 'inline')
plt.scatter(data, target, color = 'red')
plt.plot(data, pred, color = 'green')
plt.xlabel('full value property tax-rate')
plt.ylabel('Cost of House')
plt.show()


# In[14]:


#circumventing curve issue using polynomial model
from sklearn.preprocessing import PolynomialFeatures

# to allow merging of models
from sklearn.pipeline import make_pipeline


# In[15]:


model = make_pipeline(PolynomialFeatures(3), reg)


# In[16]:


# fit the model data with target
model.fit(data,target)


# In[17]:


# prediction
pred = model.predict(data)


# In[18]:


# ensuring the matplotlib is working well inside notebook
get_ipython().run_line_magic('matplotlib', 'inline')
plt.scatter(data, target, color = 'red')
plt.plot(data, pred, color = 'green')
plt.xlabel('full value property tax-rate')
plt.ylabel('Cost of House')
plt.show()


# In[19]:


# r_2 metric
from sklearn.metrics import r2_score


# In[20]:


# predict
r2_score(pred,target)


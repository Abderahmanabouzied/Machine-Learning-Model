#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np 


# In[2]:


df = pd.read_csv("student_scores.csv")


# In[3]:


df.head()


# In[4]:


df.info()


# In[10]:


X = df[["Hours"]].values
y=df["Scores"].values


# In[11]:


from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X,y ,test_size = 0.2 , random_state = 0)


# In[13]:


from sklearn import linear_model

model = linear_model.LinearRegression ()

model.fit(X_train,y_train)


# In[18]:


model.coef_


# In[14]:


y_pred = model.predict(X_test)


# In[15]:


model.predict([[3.9]])


# In[19]:


from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score 


r2_score_1 = r2_score(X_test,y_pred)
MSE = mean_squared_error(X_test,y_pred)

RMSE = np.sqrt(mean_squared_error(X_test,y_pred))


print("r2_score_1 : ", r2_score_1)
print("MSE : ", MSE)
print("RMSE : ", RMSE)


# In[23]:


import matplotlib.pyplot as plt 

plt.scatter(X,y)

plt.plot(X_test,y_pred , c = "r")


# In[35]:


from mlxtend.evaluate import bias_variance_decomp

ave_loss , ave_bais , ave_var = bias_variance_decomp(model , X_train , y_train , X_test , y_test , loss = "mse" , num_rounds = 200 , random_seed = 123 )

print("ave_loss : ", ave_loss)
print("ave_bais : ", ave_bais)
print("ave_var : ", ave_var)


# In[ ]:





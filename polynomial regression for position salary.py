#!/usr/bin/env python
# coding: utf-8

# # Polynomial Regression

# ## Importing the libraries

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# ## Importing the dataset

# In[2]:


dataset = pd.read_csv('Position_Salaries.csv')
dataset.head()


# In[3]:


X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values


# In[4]:


X


# ## Training the Linear Regression model on the whole dataset

# In[5]:


from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)


# ## Training the Polynomial Regression model on the whole dataset

# In[16]:


from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 3)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)


# ## Visualising the Linear Regression results

# In[13]:


plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()


# ## Visualising the Polynomial Regression results

# In[14]:


plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


# ## Visualising the Polynomial Regression results (for higher resolution and smoother curve)

# In[8]:


X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


# ## Predicting a new result with Linear Regression

# In[9]:


lin_reg.predict([[6.5]])


# ## Predicting a new result with Polynomial Regression

# In[10]:


lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))


# In[ ]:





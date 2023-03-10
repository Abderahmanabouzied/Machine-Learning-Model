#!/usr/bin/env python
# coding: utf-8

# # Evaluation metrics

# Evaluation metrics are used to assess the performance of a machine learning model on a given dataset. There are many different evaluation metrics, and which one to use depends on the specific task and requirements of the model.
# 
# Here are some common evaluation metrics used in machine learning, with examples of how to calculate them in Python using scikit-learn:

# --------

# ## 1-Accuracy:

# This is the most basic evaluation metric, and it simply measures the proportion of correct predictions made by the model.

# In[ ]:


from sklearn.metrics import accuracy_score

# Calculate accuracy
accuracy = accuracy_score(y_true, y_pred)


# --------

# ## 2-Precision:

# This metric measures the proportion of true positive predictions made by the model out of all positive predictions.

# In[ ]:


from sklearn.metrics import precision_score

# Calculate precision
precision = precision_score(y_true, y_pred)


# ------

# ## 3-Recall:

# This metric measures the proportion of true positive predictions made by the model out of all actual positive cases.

# In[ ]:


from sklearn.metrics import recall_score

# Calculate recall
recall = recall_score(y_true, y_pred)


# -------

# ## 4-F1 Score:

# This metric is the harmonic mean of precision and recall, and it is often used as a single measure of the overall performance of a model.

# In[ ]:


from sklearn.metrics import f1_score

# Calculate F1 score
f1 = f1_score(y_true, y_pred)


# --------

# ## Mean Absolute Error (MAE):

# The MAE is the average absolute difference between the predicted values and the true values. It is a measure of the magnitude of the error, without considering the direction.

# In[ ]:


from sklearn.metrics import mean_absolute_error

# Calculate the MAE
mae = mean_absolute_error(y_test, y_pred)
print(f'MAE: {mae}')


# ------

# ## Mean Squared Error (MSE): 

# The MSE is the average squared difference between the predicted values and the true values. It is a measure of the magnitude of the error, but it penalizes larger errors more than the MAE.

# In[ ]:


from sklearn.metrics import mean_squared_error

# Calculate the MSE
mse = mean_squared_error(y_test, y_pred)
print(f'MSE: {mse}')


# -----

# ## Root Mean Squared Error (RMSE): 

# The RMSE is the square root of the MSE. It is interpreted in the same units as the target variable, which can make it easier to understand the magnitude of the error.

# In[ ]:


from math import sqrt
from sklearn.metrics import mean_squared_error

# Calculate the RMSE
rmse = sqrt(mean_squared_error(y_test, y_pred))
print(f'RMSE: {rmse}')


# -----

# ## Mean Absolute Percentage Error (MAPE): 

# The MAPE is the average absolute error in percentage terms. It is calculated as the average absolute error divided by the average value of the target variable.

# In[ ]:


from sklearn.metrics import mean_absolute_error

# Calculate the MAPE
mape = (mean_absolute_error(y_test, y_pred) / y_test.mean()) * 100
print(f'MAPE: {mape}')


# -----

# In machine learning, the coefficient of determination, or r^2 for short, is a metric that indicates how well a model fits the data. It is a measure of the proportion of the variance in the dependent variable that is predictable from the independent variable(s).
# 
# The r^2 value ranges from 0 to 1, where a value of 0 indicates that the model does not explain any of the variance in the target variable, and a value of 1 indicates that the model explains all of the variance in the target variable. A value between 0 and 1 indicates the proportion of the variance that the model explains. The higher the r^2 value, the better the model fits the data.
# 
# Here is an example of how to calculate the r^2 value in Python using the scikit-learn library:

# In[ ]:


from sklearn.metrics import r2_score

# Calculate the r^2 value for the predictions on the test data
r2 = r2_score(y_test, y_pred)
print(f'r^2: {r2}')


# It's important to choose the appropriate evaluation metric for your specific task, as some metrics may be more important than others depending on the requirements of the model. For example, if the consequences of false negatives are more severe than false positives, recall may be more important than precision.

# In[ ]:





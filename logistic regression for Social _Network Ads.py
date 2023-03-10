#!/usr/bin/env python
# coding: utf-8

# # Logistic Regression

# ## Importing the libraries

# In[33]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# ## Importing the dataset

# In[34]:


dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, -1].values


# In[35]:


dataset.head()


# ## Splitting the dataset into the Training set and Test set

# In[36]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# ## Feature Scaling

# In[37]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# ## Training the Logistic Regression model on the Training set

# In[38]:


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)


# ## Predicting the Test set results

# In[39]:


y_pred = classifier.predict(X_test)


# ## Making the Confusion Matrix

# In[40]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("confusion_matrix \n" , cm)

acc = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(acc * 100))


# In[41]:


from sklearn.metrics import classification_report
report = classification_report(y_test, y_pred)
print(report)


# ## Visualising the Training set results

# In[12]:


from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


# ## Visualising the Test set results

# In[13]:


from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


# In[15]:


import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

# y_true and y_pred are the true labels and predicted labels, respectively
y_true = [1, 0, 1, 1, 0, 0, 1, 0, 1, 0]
y_pred = [1, 0, 0, 1, 0, 0, 1, 1, 1, 0]

# Calculate accuracy
acc = accuracy_score(y_true, y_pred)
print("Accuracy: {:.2f}%".format(acc * 100))

# Calculate confusion matrix
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)


# In[16]:


from sklearn.metrics import classification_report
print(classification_report(y_true, y_pred))


# In[42]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Load the data
data = pd.read_csv("Social_Network_Ads.csv")

# Split the data into features (X) and labels (y)
X = dataset[["Age","EstimatedSalary"]]
y =dataset ["Purchased"]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a logistic regression model
model = LogisticRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Generate a classification report
report = classification_report(y_test, y_pred)
print(report)


# In[43]:


import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

# Generate some sample data
np.random.seed(123)
y_true = np.random.randint(2, size=100)
y_scores = np.random.rand(100)

# Calculate AUC
fpr, tpr, thresholds = metrics.roc_curve(y_true, y_scores)
roc_auc = metrics.auc(fpr, tpr)

# Plot ROC curve
plt.plot(fpr, tpr, label='AUC = {:.2f}'.format(roc_auc))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='best')
plt.show()


# In[ ]:





# In[50]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import matplotlib.pyplot as plt

data = pd.read_csv("Social_Network_Ads.csv")


# In[51]:


X = dataset[["Age","EstimatedSalary"]]
y =dataset ["Purchased"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[52]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[53]:


clf = LogisticRegression()
clf.fit(X_train, y_train)


# After that, we'll make predictions on the testing data and calculate the AUC and ROC curve:

# The roc_curve function from the sklearn.metrics module is used to calculate the false positive rate (FPR) and true positive rate (TPR) at various threshold settings, and auc function is used to calculate the area under the ROC curve. Finally, the ROC curve is plotted with matplotlib.pyplot.plot, with the AUC value included in the legend.

# In[54]:


y_pred = clf.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
roc_auc = metrics.auc(fpr, tpr)


# In[55]:


plt.plot(fpr, tpr, label='AUC = {:.2f}'.format(roc_auc))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='best')
plt.show()


# This code will plot the ROC curve and show AUC score, with this plot you can decide the threshold point where you want to balance the True Positive rate and False Positive rate.

# In[ ]:





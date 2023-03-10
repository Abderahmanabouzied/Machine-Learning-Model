#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv(r"C:\Users\z0043w9p\OneDrive - Siemens Energy\Desktop\data science course\tree session\titanic_train.csv")
df


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df.isnull().sum()


# In[6]:


import seaborn as sns


# In[7]:


sns.heatmap(df.isnull(),yticklabels=False,cbar = True , cmap = "viridis")


# In[8]:


sns.set_style('whitegrid')

sns.countplot(x = "Survived", data = df ,palette = "RdBu_r" )


# In[9]:


df["Survived"].value_counts()


# In[10]:


sns.countplot(x = "Survived",hue = "Sex", data = df ,palette = "RdBu_r" )


# In[11]:


sns.countplot(x = "Survived",hue = "Pclass", data = df ,palette = "RdBu_r" )


# In[12]:


sns.distplot(df["Age"].dropna(),kde=True , color = "darkred",bins = 40)


# In[13]:


df["Age"].hist(bins = 40 , color = "darkred" , alpha= 0.9)


# In[14]:


sns.countplot(x = "SibSp", data = df ,palette = "RdBu_r" )


# In[15]:


df["Fare"].hist(bins = 60 , color = "darkred" , alpha= 0.9,figsize=(12,5))


# In[16]:


plt.figure(figsize=(12,9))
sns.boxplot(x = "Pclass", y = "Age" , data = df , palette = "RdBu_r")


# In[17]:


def im_age (col) :
    age = col[0]
    pclass = col[1]
    
    if pd.isnull(age):
        
        if pclass == 1 :
            return 37
        
        elif pclass == 2 :
            return 27
        
        else:
            return 23
    else : 
        return age


# In[18]:


df["age_new"] = df[["Age","Pclass"]].apply(im_age,axis=1)


# In[19]:


sns.heatmap(df.isnull(),yticklabels=False,cbar = True , cmap = "viridis")


# In[20]:


df[["Age","Pclass"]]


# In[21]:


df.head()


# In[ ]:





# In[22]:


df.drop("Cabin",axis = 1 ,inplace = True)


# In[23]:


df.head()


# In[24]:


sex = pd.get_dummies(df["Sex"],drop_first = True)


# In[25]:


sex


# In[26]:


df["Embarked"].unique()


# In[27]:


Embarked = pd.get_dummies(df["Embarked"],drop_first = True)


# In[28]:


Embarked


# In[29]:


df.head()


# In[30]:


sd = pd.concat([sex,Embarked],axis = 1)


# In[31]:


sd.head()


# In[32]:


df = pd.concat([df,sex,Embarked],axis = 1)


# In[33]:


df.head()


# In[34]:


df.drop(["Name","Sex","Ticket","Embarked","age_new"],axis = 1 ,inplace = True)


# In[35]:


df.head()


# In[47]:


X=df[["PassengerId"]]


# In[48]:


print(X)


# In[43]:


y=df[["Pclass"]]


# In[44]:


print(y)


# In[49]:


from sklearn.tree import DecisionTreeRegressor

model = DecisionTreeRegressor(random_state = 0, max_depth = 42 )

model.fit(X,y)


# In[50]:


from sklearn.tree import export_text

text_repo = export_text(model)
print(text_repo)


# In[51]:


from sklearn.tree import plot_tree

fig = plt.figure(figsize=(12,5),dpi=100)

visualization = plot_tree(model,feature_names = df.columns , filled = True)

print(visualization)


# In[52]:


df.columns


# In[53]:


df


# In[54]:


model.predict([[10]])


# In[57]:



plt.scatter (X,y , color = "red")

plt.plot(x_grid,model.predict(x_grid) , color = "blue")

plt.title("PassengerId")

plt.xlabel("PassengerId")

plt.ylabel("Pclass")

plt.show()


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 


# In[2]:


df = pd.read_csv("Position_Salaries.csv")


# In[3]:


df.info()


# In[4]:


df.head()


# In[5]:


X = df.iloc[:,1:-1].values
y = df.iloc[:,-1].values


# In[6]:


X


# In[7]:


y


# In[8]:


from sklearn.tree import DecisionTreeRegressor

model = DecisionTreeRegressor(random_state = 0, max_depth = 2 )

model.fit(X,y)


# In[9]:


from sklearn.tree import export_text

text_repo = export_text(model)
print(text_repo)


# In[22]:


from sklearn.tree import plot_tree

fig = plt.figure(figsize=(12,5),dpi=100)

visualization = plot_tree(model,feature_names = df.columns , filled = True)

print(visualization)


# In[12]:


df.columns


# In[25]:


df


# In[30]:


model.predict([[10]])


# In[37]:


x_grid = np.arange(min(X),max(X),0.01)
x_grid = x_grid.reshape(len(x_grid),1)

plt.scatter (X,y , color = "red")

plt.plot(x_grid,model.predict(x_grid) , color = "blue")

plt.title("DecisionTreeRegressor")

plt.xlabel("Position Level")

plt.ylabel("Salary")

plt.show()


# In[43]:


df = pd.read_csv("Carseats.csv")


# In[47]:


df.head()


# In[48]:


df.info()


# In[49]:


df.columns


# In[51]:


df["ShelveLoc"].unique()


# In[52]:


df["ShelveLoc"] = df["ShelveLoc"].map({'Bad':1, 'Good':2, 'Medium':3})


# In[54]:


df["Urban"] = df["Urban"].map({'Yes':1, 'No':2})


# In[55]:


df["US"] = df["US"].map({'Yes':1, 'No':2})


# In[56]:


df.head()


# In[58]:


X = df[['Sales', 'CompPrice', 'Income', 'Advertising', 'Population',
       'ShelveLoc', 'Age', 'Education', 'Urban', 'US']]


# In[60]:


y = df["Price"]


# In[66]:


from sklearn.tree import DecisionTreeRegressor

model = DecisionTreeRegressor(random_state = 0, max_depth = 10 )

model.fit(X,y)


# In[71]:


df.columns


# In[73]:


from sklearn.tree import export_text
columns = ['Sales', 'CompPrice', 'Income', 'Advertising', 'Population',
       'ShelveLoc', 'Age', 'Education', 'Urban', 'US']
text_repo = export_text(model,feature_names =columns )
print(text_repo)


# In[70]:


from sklearn.tree import plot_tree

fig = plt.figure(figsize=(300,100),dpi=100)

visualization = plot_tree(model,feature_names = df.columns , filled = True)

print(visualization)


# In[65]:


model.predict([[3,180,40,20,89,2,18,28,1,2]])


# In[ ]:





# In[64]:


df.head()


# In[2]:


df = pd.read_csv("titanic_train.csv")


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df.isnull().sum()


# In[6]:


891-687


# In[7]:


import seaborn as sns


# In[8]:


sns.heatmap(df.isnull(),yticklabels=False,cbar = True , cmap = "viridis")


# In[9]:


sns.set_style('whitegrid')

sns.countplot(x = "Survived", data = df ,palette = "RdBu_r" )


# In[10]:


df["Survived"].value_counts()


# In[11]:


sns.countplot(x = "Survived", data = df ,palette = "RdBu_r" )


# In[12]:


sns.countplot(x = "Survived",hue = "Sex", data = df ,palette = "RdBu_r" )


# In[13]:


sns.countplot(x = "Survived",hue = "Pclass", data = df ,palette = "RdBu_r" )


# In[14]:


sns.distplot(df["Age"].dropna(),kde=True , color = "darkred",bins = 40)


# In[15]:


df["Age"].hist(bins = 40 , color = "darkred" , alpha= 0.9)


# In[16]:


sns.countplot(x = "SibSp", data = df ,palette = "RdBu_r" )


# In[17]:


df["Fare"].hist(bins = 60 , color = "darkred" , alpha= 0.9,figsize=(12,5))


# In[18]:


plt.figure(figsize=(12,9))
sns.boxplot(x = "Pclass", y = "Age" , data = df , palette = "RdBu_r")


# In[19]:


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
    


# In[20]:


df["age_new"] = df[["Age","Pclass"]].apply(im_age,axis=1)


# In[21]:


sns.heatmap(df.isnull(),yticklabels=False,cbar = True , cmap = "viridis")


# In[22]:


df[["Age","Pclass"]]


# In[23]:


df.head()


# In[24]:


df.drop("Cabin",axis = 1 ,inplace = True)


# In[25]:


df.head()


# In[26]:


sex = pd.get_dummies(df["Sex"],drop_first = True)


# In[27]:


sex


# In[28]:


df["Embarked"].unique()


# In[29]:


Embarked = pd.get_dummies(df["Embarked"],drop_first = True)


# In[30]:


Embarked


# In[31]:


df.head()


# In[32]:


sd = pd.concat([sex,Embarked],axis = 1)


# In[34]:


sd.head()


# In[35]:


df = pd.concat([df,sex,Embarked],axis = 1)


# In[36]:


df.head()


# In[37]:


df.drop(["Name","Sex","Ticket","Embarked","age_new"],axis = 1 ,inplace = True)


# In[38]:


df.head()


# In[ ]:





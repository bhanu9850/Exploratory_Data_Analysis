#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd 


# In[3]:


df  = pd.read_csv("synthetic_ml_dataset.csv")


# In[4]:


df


# In[5]:


df.head()


# In[6]:


df.describe()


# In[7]:


df.shape


# In[8]:


df.info()


# In[9]:


df.isnull().sum()


# In[10]:


df.dtypes


# In[11]:


df['Age'] = pd.to_numeric(df['Age'],errors = 'coerce')


# In[12]:


df['Age'].fillna(df['Age'].mean(),inplace = True)


# In[13]:


df['Age'].isnull().sum()


# In[14]:


df['Joining_Date'] = pd.to_datetime(df['Joining_Date'],errors = 'coerce')


# In[15]:


df['Joining_Date'].isnull().sum()


# In[16]:


df['Joining_Date'].fillna(method = "ffill",inplace = True)


# In[17]:


df['Joining_Date'].isnull().sum()


# In[18]:


df['Gender'].fillna(df["Gender"].mode()[0],inplace = True )
df['City'].fillna(df["City"].mode()[0],inplace = True )
df['Department'].fillna(df["Department"].mode()[0],inplace = True )


# In[19]:


df['Salary'].fillna(df['Salary'].mean(),inplace = True)
df['Remote_Work'].fillna(False,inplace = True )
df['Performance_Score'].fillna(df['Performance_Score'].mean(),inplace = True )
df.dropna(subset = ['Name'],inplace = True)


# In[20]:


df.info()


# In[21]:


df.head()


# In[22]:


df = pd.get_dummies(df,columns = ['Gender','City','Department','Remote_Work'],drop_first=True)


# In[23]:


df


# In[24]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score


# In[25]:


x = df.drop(['Performance_Score','Name','Joining_Date'],axis = 1)
y = df['Performance_Score']
print(x)
print(y)


# In[26]:


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# In[27]:


model = RandomForestRegressor()
model.fit(X_train,y_train)


# In[28]:


y_pred = model.predict(X_test)


# In[29]:


y_pred


# In[30]:


print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))
print("R² Score:", r2_score(y_test, y_pred))


# In[ ]:





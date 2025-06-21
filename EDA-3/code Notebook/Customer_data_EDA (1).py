#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")


# In[4]:


df = pd.read_csv('customer_dataset.csv')
df


# In[5]:


df.info()


# In[6]:


df.shape


# In[7]:


df.describe()


# In[8]:


df.isnull().sum()


# In[9]:


df.dtypes


# In[10]:


df['Age'] = pd.to_numeric(df['Age'],errors = 'coerce')
df['Signup_Date'] = pd.to_datetime(df['Signup_Date'],errors = 'coerce')


# In[11]:


df.dtypes


# In[12]:


df['Age'].fillna(df['Age'].median(), inplace=True)
df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
df['City'].fillna(df['City'].mode()[0], inplace=True)

df['Membership_Level'].fillna(method='bfill', inplace=True)

df['Last_Purchase_Amount'].fillna(df['Last_Purchase_Amount'].mean(), inplace=True)
df['Purchase_Frequency'].fillna(df['Purchase_Frequency'].mean(), inplace=True)

df['Churn_Status'].fillna(method='bfill', inplace=True)


# In[13]:


df.isnull().sum()


# In[14]:


df['Churn_Status'].fillna('Yes', inplace=True)


# In[15]:


df = df.dropna(subset=['Name'])


# In[16]:


df.shape


# In[17]:


df.isnull().sum()


# In[18]:


df.head()


# In[19]:


df.duplicated().sum()


# In[20]:


df.to_csv("preprocessed_customer_dataset.csv", index=False)


# In[21]:


plt.figure(figsize=(8, 5))
sns.histplot(df['Age'], bins=30, kde=True, color='skyblue')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()


# In[22]:


plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Churn_Status', hue='Gender', palette='coolwarm')
plt.title('Churn Status by Gender')
plt.show()


# In[23]:


plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Gender', palette='Set2')
plt.title('Gender Distribution')
plt.show()


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
import matplotlib.pyplot as plt


# In[2]:


#Taking dataset as input
input_file = pd.read_csv("D:\Downloads\engineering colleges in India (1).csv", low_memory=False)


# In[3]:


input_file.head()


# In[4]:


input_file.info()


# In[5]:


#Creating a new column called student_to_faculty_ratio
students="Total Student Enrollments"
teachers="Total Faculty"


# In[6]:


input_file[students] = pd.to_numeric(input_file[students], errors='coerce')
input_file[teachers] = pd.to_numeric(input_file[teachers], errors='coerce')


# In[7]:


input_file["student_to_faculty_ratio"]=input_file[students]/input_file[teachers]


# In[8]:


input_file.head()


# In[9]:


input_file["student_to_faculty_ratio"]


# In[10]:


input_file.info()


# In[11]:


input_file.describe()


# In[12]:


# Check the columns that contain 'Unnamed'
cols_to_drop = input_file.columns[input_file.columns.str.contains('^Unnamed')]

# Drop the columns from the DataFrame
input_file.drop(columns=cols_to_drop, inplace=True)


# In[13]:


input_file.head()


# In[14]:


input_file.describe()


# In[15]:


input_file.info()


# In[16]:


#Creating a copy of the input_file dataset with necessary features 
features=["College Name","student_to_faculty_ratio","University","Facilities","City","College Type","Rating"]
data_set=input_file[features].copy()


# In[17]:


data_set.head()


# In[18]:


data_set.info()


# In[19]:


#Dropping the unnecessary columns
feature_drop=["University","College Name"]
data_set.drop(columns=["University"], inplace=True)
data_set.drop(columns=["College Name"], inplace=True)
data_set.head()


# In[20]:


#Encoding the categorical column - College Type
from sklearn.preprocessing import LabelEncoder

one_hot_encoded = pd.get_dummies(data_set, columns=['College Type'])

# Label Encoding
dataset = data_set.copy()  # Make a copy of the DataFrame
label_encoder = LabelEncoder()
dataset['College Type'] = label_encoder.fit_transform(dataset['College Type'])


# In[59]:


#Create a separate train dataset that contains the rows in which rating is present
dataset = dataset.dropna(subset=['Rating'])


# In[60]:


dataset.head()


# In[64]:


# Make a copy of the dataset
dataset_t = dataset.copy()
# Drop unnecessary columns
dataset_t=dataset_t.drop(columns=['Facilities','City'])
# Drop rows with missing values in the 'rating' column
dataset_t.dropna(subset=['Rating'])
dataset_t.info()


# In[67]:


X = dataset_t.drop(columns=['Rating'])
y = dataset_t['Rating']
X


# In[68]:


X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Drop samples with missing values
X_train_cleaned = X_train.dropna()

# Drop corresponding labels in y_train as well if there are missing values
y_train_cleaned = y_train.drop(y_train.index.difference(X_train_cleaned.index))

X_valid_cleaned = X_valid.dropna()

# Drop corresponding labels in y_train as well if there are missing values
y_valid_cleaned = y_valid.drop(y_valid.index.difference(X_valid_cleaned.index))



# In[69]:


print("Training dataset feature names:", X_train_cleaned.columns)
X_train_cleaned


# In[70]:


#Using RandomForestRegressor to predict rating
model = RandomForestRegressor(random_state=42)
model.fit(X_train_cleaned, y_train_cleaned)
X_valid_cleaned = X_valid.dropna()
y_valid_cleaned = y_valid.drop(y_valid.index.difference(X_valid_cleaned.index))

# making predictions on the cleaned validation set
y_pred = model.predict(X_valid_cleaned)

# Calculate RMSE (Root Mean Squared Error) to evaluate the model performance
rmse = sqrt(mean_squared_error(y_valid_cleaned, y_pred))
print(f"RMSE: {rmse}")


# In[71]:


#Creating a output dataset with rows that doesn't have rating to predict the rating
dataset_main_subtracted = data_set.drop(dataset.index, errors='ignore')


# In[72]:


dataset_main=dataset_main_subtracted.drop(columns=['Facilities','City'])


# In[73]:


dataset_main


# In[74]:


X_test=dataset_main.drop(columns='Rating')


# In[75]:


X_test


# In[76]:


#Encoding College_type
from sklearn.preprocessing import LabelEncoder

one_hot_encoded = pd.get_dummies(X_test, columns=['College Type'])

# Label Encoding
xtest = X_test.copy()  # Make a copy of the DataFrame
label_encoder = LabelEncoder()

xtest['College Type'] = label_encoder.fit_transform(X_test['College Type'])


# In[77]:


xtest.dropna(subset=['student_to_faculty_ratio'])


# In[78]:


xtest.dropna(subset=['College Type'])
xtest


# In[80]:


xtest=xtest.dropna(subset=['student_to_faculty_ratio'])
xtest


# In[81]:


missing_values = xtest.isnull().sum()
print(missing_values)


# In[82]:


# making predictions for rating
predictions=model.predict(xtest)


# In[83]:


predictions


# In[86]:


#adding the predictions to the test dataset as a column
xtest.loc[:, "Rating"] = predictions


# In[87]:


xtest


# In[88]:


missing_values = input_file.isnull().sum()
print(missing_values)


# In[89]:


#Combining the input_file dataset with the dataset that has the predictions 
combined_df = xtest.merge(input_file, left_index=True, right_index=True)


# In[90]:


combined_df


# In[92]:


combined_df.head(1)


# In[93]:


# Converting the final output file to a csv file
combined_df.to_csv('output_file.csv', index=False)


# In[94]:


#Creating a link to the output file
from IPython.display import FileLink

FileLink('output_file.csv')


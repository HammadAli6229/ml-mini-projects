#!/usr/bin/env python
# coding: utf-8

# # Housing Price Prediction (Linear Regression)
# Implemented a linear regression model to predict the prices of houses based on their square footage and the number of bedrooms and bathrooms. Using Kaggle Dataset of 'Housing'.

# In[1]:


# Supress Warnings
import warnings
warnings.filterwarnings('ignore')

# Import the numpy and pandas package
import numpy as np
import pandas as pd

# Data Visualisation
import matplotlib.pyplot as plt 
import seaborn as sns

# Importing necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import r2_score
from IPython.display import display, HTML


# In[2]:


# Load the dataset
housing = pd.read_csv("Housing.csv")


# In[3]:


# Check the head of the dataset
print("First few rows of the dataset:")
display(housing.head())


# In[4]:


# Check the shape of the dataset
print("\nShape of the dataset:")
print(housing.shape)


# In[5]:


# Check the information about the dataset
print("\nInformation about the dataset:")
df_info = housing.info()
display(HTML(df_info))


# In[6]:


# Check the summary statistics of the dataset
print("\nSummary statistics of the dataset:")
display(housing.describe())


# In[7]:


# Checking Null values
print("\nPercentage of null values in each column:")
null_values = housing.isnull().sum() * 100 / housing.shape[0]
print(null_values)
# There are no NULL values in the dataset, hence it is clean.


# In[8]:


# Outlier Analysis
print("\nOutlier analysis for the dataset:")
plt.figure(figsize=(10, 5))
sns.boxplot(data=housing)
plt.title('Boxplot of all features')
plt.show()


# In[9]:


# Outlier treatment for price
Q1 = housing['price'].quantile(0.25)
Q3 = housing['price'].quantile(0.75)
IQR = Q3 - Q1
housing = housing[(housing['price'] >= Q1 - 1.5*IQR) & (housing['price'] <= Q3 + 1.5*IQR)]

# Outlier treatment for area
Q1 = housing['area'].quantile(0.25)
Q3 = housing['area'].quantile(0.75)
IQR = Q3 - Q1
housing = housing[(housing['area'] >= Q1 - 1.5*IQR) & (housing['area'] <= Q3 + 1.5*IQR)]


# In[10]:


# Check the shape of the dataset after outlier treatment
print("\nShape of the dataset after outlier treatment:")
print(housing.shape)


# In[11]:


# Data Visualization
print("\nPairplot for the dataset:")
plt.figure(figsize=(20, 12))
sns.pairplot(housing)
plt.show()


# In[12]:


# Convert categorical variables to binary
print("\nConverting categorical variables to binary:")
varlist = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
housing[varlist] = housing[varlist].apply(lambda x: x.map({'yes': 1, 'no': 0}))


# In[13]:


# Get dummy variables for 'furnishingstatus'
status = pd.get_dummies(housing['furnishingstatus'], drop_first=True)

# Concatenate status with original dataframe
housing = pd.concat([housing, status], axis=1)

# Drop 'furnishingstatus' column
housing.drop(['furnishingstatus'], axis=1, inplace=True)


# In[14]:


# Display the transformed dataset
print("\nTransformed dataset:")
display(housing.head())


# In[15]:


# Train-test split
np.random.seed(0)
df_train, df_test = train_test_split(housing, train_size=0.7, test_size=0.3, random_state=100)

# Scale the numerical features
scaler = MinMaxScaler()
num_vars = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking', 'price']
df_train[num_vars] = scaler.fit_transform(df_train[num_vars])

# Select features and target variable for training
X_train = df_train[['area', 'bedrooms', 'bathrooms']]
y_train = df_train['price']


# In[16]:


# Fit linear regression model
print("\nFitting linear regression model:")
lm = LinearRegression()
lm.fit(X_train, y_train)


# In[17]:


# Make predictions on training set
y_train_pred = lm.predict(X_train)


# In[18]:


# R-squared score on training set
r2_train = r2_score(y_train, y_train_pred)
print("R-squared score on training set:", r2_train)


# In[19]:


# Check for multicollinearity
print("\nChecking for multicollinearity:")
correlation_matrix = np.corrcoef(X_train.values.T)
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', xticklabels=X_train.columns, yticklabels=X_train.columns)
plt.title('Correlation Matrix of Features')
plt.show()


# In[20]:


# Test set preparation
df_test[num_vars] = scaler.transform(df_test[num_vars])
X_test = df_test[['area', 'bedrooms', 'bathrooms']]
y_test = df_test['price']

# Make predictions on test set
y_test_pred = lm.predict(X_test)


# In[21]:


# R-squared score on test set
r2_test = r2_score(y_test, y_test_pred)
print("R-squared score on test set:", r2_test)


# In[22]:


# Plot y_test vs y_pred
print("\nPlotting y_test vs y_pred:")
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_test_pred, c='blue', label='Actual vs Predicted')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='Ideal Linear Relationship')
plt.title('Actual vs Predicted Values')
plt.xlabel('Actual (y_test)')
plt.ylabel('Predicted (y_pred)')
plt.legend()
plt.show()


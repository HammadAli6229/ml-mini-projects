#!/usr/bin/env python
# coding: utf-8

# # Customer Segmentation (K-Mean Clustering)
# We used K-means clustering algorithm to group customers of a retail store based on their purchase history.

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from IPython.display import display, HTML


# In[2]:


# Load the data
df = pd.read_csv("Mall_Customers.csv")


# In[3]:


# Data exploration
print("Data Exploration:")
display(df.head())
print("\nShape of the dataset:", df.shape)
print("\nInfo about the dataset:")
df_info = df.info()
display(HTML(df_info))
print("\nNumber of duplicated rows:", df.duplicated().sum())
print("\nDescriptive statistics:")
display(df.describe())


# ### Data Exploration and Visualization
# Plot pairwise relationships between features in a dataset.

# In[4]:


# Visualizing data distributions
plt.figure(figsize=(16, 10))
sns.pairplot(data=df, hue='Gender')
plt.suptitle('Pairplot of Features by Gender', y=1.02, fontsize=16)
plt.show()


# In[5]:


plt.figure(figsize=(4, 4))
sns.countplot(x='Gender', data=df)
plt.title('Gender Distribution')
plt.show()


# ### Distribution of numerical features (Age, Annual income & Spending score)

# In[6]:


plt.figure(figsize=(16, 4))
for i, col in enumerate(['Age', 'Annual Income (k$)', 'Spending Score (1-100)']):
    plt.subplot(1, 3, i + 1)
    sns.histplot(df[col], bins=10, kde=True)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
plt.tight_layout()
plt.show()


# ### Clustering using K-means (ML Model)
# #### Application in this use-case:
# Let's perform clustering (optimizing K with the elbow method). In order to simplify the problem, we start by keeping only the two last columns as features.
# ##### Optimal K: the elbow method
# How many clusters would you choose ?
# 
# A common, empirical method, is the elbow method. You plot the mean distance of every point toward its cluster center, as a function of the number of clusters.
# 
# Sometimes the plot has an arm shape, and the elbow would be the optimal K.

# In[7]:


# Clustering
X = df.iloc[:, -3:]


# In[8]:


# Elbow Method to find optimal k
inertias = []
silhouette_scores = []
for k in range(2, 11):
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X)
    inertias.append(km.inertia_)
    silhouette_scores.append(silhouette_score(X, km.labels_))

plt.figure(figsize=(12, 6))
plt.plot(range(2, 11), inertias, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.xticks(range(2, 11))
plt.grid(True)
plt.show()


# In[9]:


plt.figure(figsize=(12, 6))
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.title('Silhouette Score for Optimal k')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.xticks(range(2, 11))
plt.grid(True)
plt.show()


# In[10]:


# Performing KMeans with optimal k
optimal_k = 5
km = KMeans(n_clusters=optimal_k, random_state=42)
df['Cluster'] = km.fit_predict(X)


# In[11]:


# Visualizing Clusters
plt.figure(figsize=(12, 8))
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', data=df, palette='viridis', legend='full')
plt.title('Clusters of Customers based on Purchase History', fontsize=16)
plt.xlabel('Annual Income (k$)', fontsize=14)
plt.ylabel('Spending Score (1-100)', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(title='Cluster', fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()


# In[12]:


# Analyzing Clusters
cluster_centers = pd.DataFrame(km.cluster_centers_, columns=X.columns)
cluster_centers['Cluster'] = cluster_centers.index
cluster_centers['Cluster'] = cluster_centers['Cluster'].apply(lambda x: f'Cluster {x}')


# In[13]:


print("Cluster Centers:")
display(cluster_centers)


# ### Definition of customers profiles corresponding to each clusters

# In[14]:


# Number of customers in each cluster
cluster_sizes = df['Cluster'].value_counts().sort_index()
print("\nNumber of Customers in Each Cluster:")
display(cluster_sizes)


# In[15]:


# Visualizing cluster centers in tabular form
plt.figure(figsize=(10, 6))
sns.heatmap(cluster_centers.set_index('Cluster'), annot=True, cmap='viridis', fmt='.2f', linewidths=0.5)
plt.title('Cluster Centers', fontsize=16)
plt.xlabel('Features', fontsize=14)
plt.ylabel('Cluster', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()


# In[16]:


# Visualizing number of customers in each cluster
plt.figure(figsize=(8, 6))
sns.barplot(x=cluster_sizes.index, y=cluster_sizes.values, palette='viridis')
plt.title('Number of Customers in Each Cluster', fontsize=16)
plt.xlabel('Cluster', fontsize=14)
plt.ylabel('Number of Customers', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()


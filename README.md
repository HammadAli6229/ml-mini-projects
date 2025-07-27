# 🧠 Machine Learning Projects

This repository contains two complete machine learning projects demonstrating core concepts in **unsupervised learning (clustering)** and **supervised learning (regression)** using real-world datasets.

## 📁 Projects Overview

1. **Customer Segmentation** – Using K-Means Clustering  
2. **Housing Price Prediction** – Using Linear Regression

---

## 🔹 Project 1: Customer Segmentation (K-Means Clustering)

### 📌 Objective
Segment customers based on their demographics and spending behavior to help businesses with targeted marketing strategies.

### 🧰 Key Steps

- **Data Loading & Exploration**
  - Load CSV file
  - Check shape, types, duplicates, and summary statistics
- **Data Visualization**
  - Gender distribution
  - Pairplots and histograms of features
- **Clustering**
  - Elbow Method & Silhouette Score to find optimal `k`
  - Apply **K-Means** and assign cluster labels
- **Cluster Analysis**
  - Scatter plot (`Annual Income` vs `Spending Score`)
  - Cluster center heatmap and distribution

### 📊 Tools Used
`Pandas`, `Matplotlib`, `Seaborn`, `Scikit-learn`

---

## 🔹 Project 2: Housing Price Prediction (Linear Regression)

### 📌 Objective
Predict housing prices based on property features such as area, number of bedrooms, and furnishing status.

### 🧰 Key Steps

- **Data Loading & Cleaning**
  - Load CSV, handle null values
  - Outlier detection & treatment using IQR
- **Exploratory Data Analysis (EDA)**
  - Boxplots, pairplots, summary statistics
- **Data Preprocessing**
  - Convert categorical features to dummy variables
  - Scale features using `MinMaxScaler`
- **Modeling**
  - Train-test split
  - Train **Linear Regression** model
  - Evaluate using **R² score**
- **Diagnostics**
  - Correlation heatmap for multicollinearity
  - Actual vs Predicted plot

### 📊 Tools Used
`Pandas`, `NumPy`, `Matplotlib`, `Seaborn`, `Scikit-learn`

---

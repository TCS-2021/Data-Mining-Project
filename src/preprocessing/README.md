# 📊 Data Preprocessing Tool

## 🚀 Overview
To assist users in cleaning and preparing their data before applying descriptive, predictive, and prescriptive analytics.

- Handling of missing values using various imputation strategies
- Data smoothing using techniques like Moving Average, Exponential, Gaussian, and LOESS
- Outlier detection and treatment
- Variance analysis and feature selection
- Option to export the final preprocessed dataset

---

## 🔧 Features

### 1. **Missing Values Handling**
Choose from:
- Mean / Median / Mode Imputation
- Regression Imputation
- Decision Tree Imputation
- Drop rows with missing values

### 2. **Smoothing**
Apply smoothing to numeric columns using:
- Moving Average
- Exponential Weighted Average
- Gaussian Smoothing
- LOESS (Locally Estimated Scatterplot Smoothing)

### 3. **Outlier Handling**
Detect and treat outliers using:
- IQR Method
- Z-Score
- Modified Z-Score
- Percentile-Based Filtering

Treatment Options:
- Remove outliers
- Cap values
- Replace with median

### 4. **Variance Analysis**
Analyze and select features using:
- Variance Threshold
- Top-N Variance
- Manual Feature Selection

Also includes correlation heatmap and downloadable dataset with selected features.

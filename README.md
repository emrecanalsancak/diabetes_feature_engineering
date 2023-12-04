# Diabetes Prediction with Feature Engineering and Random Forest Classifier

This repository contains Python code for predicting diabetes using a Random Forest Classifier. The analysis includes data preprocessing, feature engineering, and model training and evaluation.

## Dataset

The dataset contains information on various health-related features and an "Outcome" column indicating the presence (1) or absence (0) of diabetes. The variables are:

- **Pregnancies:** Number of pregnancies
- **Glucose:** 2-hour plasma glucose concentration during an oral glucose tolerance test
- **Blood Pressure:** Diastolic blood pressure (mm Hg)
- **Skin Thickness:** Triceps skinfold thickness (mm)
- **Insulin:** 2-hour serum insulin (mu U/ml)
- **BMI:** Body Mass Index
- **Age:** Age in years
- **Outcome:** Presence (1) or absence (0) of diabetes

## Analysis Steps

1. **Data Loading and Initial Exploration:**
   - Load the dataset and explore basic information.

2. **Data Cleaning and Exploration:**
   - Check for data quality and distribution.

3. **Feature Selection and Analysis:**
   - Identify and categorize variables into types.

4. **Variable Analysis:**
   - Generate summaries and visualizations for categorical and numerical variables.

5. **Target Summary with Numerical Variables:**
   - Display mean of numerical variables grouped by the binary target variable.

6. **Handling Missing Values:**
   - Impute missing values using K-Nearest Neighbors (KNN) imputation.

7. **Outlier Detection and Treatment:**
   - Identify and replace outliers using the interquartile range (IQR) method.

8. **Correlation Analysis:**
   - Visualize the correlation matrix using a heatmap.

9. **Feature Engineering:**
   - Create new variables based on age, BMI, glucose, and interactions between existing variables.

10. **Encoding and Standardization:**
    - Encode categorical variables and standardize numerical variables.

11. **Modeling:**
    - Split the dataset, train a Random Forest Classifier, and evaluate performance.

## Results
The Random Forest Classifier achieves an accuracy of 78% after feature engineering, indicating improved performance compared to the initial accuracy of 75%.

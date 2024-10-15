# 🚬 Smoking Detection Using Health Attributes - Machine Learning Project

## Project Overview

This project aims to predict smoking behavior (smoker vs. non-smoker) using health-related attributes through various machine learning models. By utilizing features such as BMI, ALT, AST, GTP, blood sugar levels, and more, the models attempt to classify individuals based on their likelihood of being a smoker. I have employed multiple classifiers and leveraged **SMOTE** to handle class imbalance, focusing on improving the model's ability to detect smokers.

## Key Features
- **Health-based Classification**: Predict whether a person smokes using health metrics.
- **Class Imbalance Handling**: Applied **SMOTE** to handle the imbalance between smokers and non-smokers.
- **Model Evaluation**: Compare multiple machine learning models and analyze the impact of SMOTE.

## Dataset

The dataset contains numerous health-related attributes for each individual. The target variable `smoking` is binary:
- `0` - Non-Smoker
- `1` - Smoker

### Key Features:
- **bmi**
- **systolic**
-  **ALT**
-   **GTP**
-    **AST**
- **blood sugar level**
- **cholesterol**
- **LDL, HDL**, and more health-related metrics.

### Objective:
To develop a model that can accurately predict whether an individual is a smoker or non-smoker based on these health metrics.

## Models Used

I have experimented with the following machine learning models:
1. **Logistic Regression**
2. **XGBoost**
3. **LightGBM**
4. **Random Forest** (Best performing model with SMOTE)

### Handling Class Imbalance
Since the dataset has more non-smokers than smokers, i used **SMOTE** (Synthetic Minority Over-sampling Technique) to create synthetic samples for the minority class (smokers). This helps to balance the dataset and improve the model's ability to detect smokers.

## Model Evaluation

The models were evaluated using the following metrics:
- **Precision**: Proportion of true positives out of predicted positives.
- **Recall**: Proportion of true positives out of actual positives.
- **F1-Score**: Harmonic mean of precision and recall.
- **Confusion Matrix**: To visualize performance on each class.
- **ROC-AUC Score**: Measures the overall performance of the classification model.

## Results

The **Random Forest Classifier with SMOTE** performed the best overall in this project, showing a balanced improvement in detecting smokers (the minority class) without compromising performance on non-smokers.

### Key Metrics of the Best Model (Random Forest with SMOTE):
- **Precision (Class 0 - Non-Smoker)**: 0.90
- **Recall (Class 1 - Smoker)**: 0.84
- **Accuracy**: 0.82

This model effectively handled class imbalance, significantly improving recall for smokers, which is crucial for real-world applications where catching smokers is more important.

## Installation and Setup

To replicate this project, you’ll need to install the following dependencies:

```bash
pip install scikit-learn
pip install lightgbm
pip install xgboost
pip install imbalanced-learn
pip install shap
```

### Steps to Run the Project:
1. **Clone the Repository**:

```bash
git clone https://github.com/your-username/your-repository.git
cd your-repository
```

2. **Preprocess the Data**:
   - Load the dataset.
   - Scale the features and apply **SMOTE** for balancing the classes if needed.

3. **Train the Models**:
   - Train models like Logistic Regression, Random Forest, XGBoost, and LightGBM.
   - Optionally apply **SMOTE** to balance the training data.

4. **Evaluate the Models**:
   - Use precision, recall, F1-score, ROC-AUC, and confusion matrix to evaluate the performance.
   - Compare the performance of each model with and without SMOTE.

5. **Analyze Results**:
   - Based on the evaluation, the **Random Forest Classifier with SMOTE** was the best-performing model for this dataset.

## Conclusion

In this project, I explored several machine learning techniques to predict smoking behavior based on health attributes. After applying **SMOTE**, the **Random Forest Classifier** emerged as the best model, particularly for improving recall on the minority class (smokers). This project demonstrates how balancing class distribution and choosing the right model can significantly improve prediction accuracy.

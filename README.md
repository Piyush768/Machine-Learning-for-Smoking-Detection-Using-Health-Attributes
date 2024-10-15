# Machine Learning for Smoking Detection Using Health Attributes

## Project Overview

This project aims to predict whether an individual is a smoker or non-smoker based on various health attributes such as waist circumference, BMI, blood sugar levels, and more. 
Multiple machine learning classifiers were employed, including Logistic Regression, XGBoost, LightGBM, and Random Forest. The effect of applying **SMOTE** (Synthetic Minority Over-sampling Technique) to handle class imbalance was also analyzed.

## Dataset

The dataset contains health-related attributes with the target variable `smoking`:
- `0` - Non-Smoker
- `1` - Smoker

Key attributes include:
- `waist(cm)`
- `bmi`
- `systolic`
- `fasting blood sugar`
- `cholesterol`
- Other health-related metrics

## Models Used

The following machine learning models were utilized:
1. **Logistic Regression**
2. **XGBoost**
3. **LightGBM**
4. **Random Forest**

### Handling Class Imbalance

To address the class imbalance, **SMOTE** was applied to oversample the minority class (smokers) in the training data.

## Model Evaluation

Each model was evaluated using:
- **Precision**: Accuracy of positive predictions.
- **Recall**: Ability to capture all positive instances.
- **F1-score**: Harmonic mean of precision and recall.
- **Confusion Matrix**
- **ROC-AUC Score**

### Best Performing Model

- **With SMOTE**: The **Random Forest Classifier with SMOTE** performed best overall, significantly improving recall for the minority class (smokers), while maintaining a strong performance for non-smokers.

## Installation

To run this project, ensure you have the following packages installed:

```bash
pip install scikit-learn
pip install lightgbm
pip install xgboost
pip install imbalanced-learn
pip install shap
```

## How to Run

1. Clone the repository:

```bash
git clone https://github.com/your-username/your-repository.git
cd your-repository
```

2. Load the dataset and preprocess the data (including scaling and SMOTE application if needed).
3. Train the machine learning models.
4. Evaluate the models based on classification metrics.
5. Analyze the impact of SMOTE on model performance.

## Results

- The **Random Forest Classifier with SMOTE** performed the best, particularly by improving recall for the smoker class while maintaining balanced performance across both classes.

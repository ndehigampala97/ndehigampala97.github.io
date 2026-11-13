# Nipuni Dehigampala – Data Science Projects

**Ph.D. Student in Chemistry | University of Florida**

Welcome to my data science project portfolio! I am a Chemistry Ph.D. student at the University of Florida, interested in applying computational methods, machine learning, and artificial intelligence to scientific problems.

---

## Project 1: Breast Cancer Diagnosis Prediction Using Machine Learning

### The Question
Can machine learning algorithms accurately classify breast tumors as benign (noncancerous) or malignant (cancerous) using characteristics of cell nuclei?

### The Dataset
This project uses the **Wisconsin Diagnostic Breast Cancer (WDBC) dataset** from the UCI Machine Learning Repository.

- **569 samples:** 357 benign and 212 malignant
- **30 numerical features** describing cell nuclei
- Features include radius, texture, perimeter, area, smoothness, compactness, concavity, and symmetry.

### What I Did
1. Loaded and explored the breast cancer dataset using Pandas and NumPy.
2. Performed exploratory data analysis using histograms, boxplots, and correlation heatmaps.
3. Prepared the data using an 80/20 stratified training-test split.
4. Developed four machine learning classification models:
   - Logistic Regression
   - Random Forest
   - Gradient Boosting
   - XGBoost
5. Evaluated the models using five-fold stratified cross-validation.
6. Optimized model hyperparameters using GridSearchCV.
7. Compared model performance using accuracy, precision, recall, F1-score, and ROC-AUC.
8. Analyzed confusion matrices, ROC curves, and feature importance.

### What I Found

All four machine learning models demonstrated strong classification performance.

| Model | Test Accuracy | ROC-AUC |
|---|---|---|
| Logistic Regression | 96.49% | 0.9960 |
| Random Forest | 97.37% | 0.9929 |
| Gradient Boosting | 96.49% | 0.9931 |
| XGBoost | 96.49% | 0.9911 |

**Key findings:**
- Random Forest achieved the highest test accuracy (97.37%).
- Logistic Regression achieved the highest test ROC-AUC (0.9960).
- Logistic Regression and Random Forest each correctly identified 39 of the 42 malignant tumors in the test set.
- Features related to tumor size and nuclear irregularity, particularly worst area and worst concave points, were important predictors in the Random Forest model.

These results demonstrate the potential of machine learning for distinguishing benign and malignant tumors. However, additional validation would be required before clinical application.

### Project Notebook

The complete Jupyter notebook contains the Python code, data visualizations, model training, hyperparameter tuning, and performance evaluation.

### Tools and Libraries
Python, Jupyter Notebook, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, and XGBoost.

**Dataset source:** [UCI Breast Cancer Wisconsin (Diagnostic) Dataset](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)

---

## About Me

I am a 2nd year Ph.D. student in Chemistry at the University of Florida, working in the prof. Pérez Research Group. My research focuses on computational chemistry, molecular simulations, protein modeling, and artificial intelligence-driven approaches to biological problems.

Through ECH6845 Chemical Process Data Science, I am developing skills in data analysis, visualization, and machine learning to support scientific research.

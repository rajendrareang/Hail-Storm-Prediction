# Hail-Storm-Prediction

## Overview
This project aims to predict hail storms using machine learning techniques. It utilizes various Python libraries and machine learning algorithms to analyze weather data and make predictions about the likelihood of hail storms.

## Libraries Used

- `numpy` (imported as `np`): For numerical computations and array manipulation.
- `pandas` (imported as `pd`): For data manipulation and analysis.
- `matplotlib.pyplot` (imported as `plt`): For data visualization.
- `seaborn` (imported as `sb`): For enhanced data visualization.
- `joblib`: For saving and loading trained machine learning models.
- `sklearn.metrics`: For evaluating model performance using metrics such as confusion matrix.
- `sklearn.model_selection.train_test_split`: For splitting the dataset into training and testing sets.
- `sklearn.preprocessing.StandardScaler`: For standardizing feature values.
- `sklearn.svm.SVC`: Support Vector Classifier for classification tasks.
- `xgboost.XGBClassifier`: XGBoost Classifier, a powerful gradient boosting algorithm.
- `sklearn.linear_model.LogisticRegression`: Logistic Regression model for classification tasks.
- `imblearn.over_sampling.RandomOverSampler`: For handling class imbalance by oversampling the minority class.

## Setup

To run the project, make sure you have Python installed on your system along with the necessary libraries listed above. You can install the required libraries using pip:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost imbalanced-learn 
```

## Usage
Clone the repository to your local machine.
Ensure you have the necessary dataset (if any) in the appropriate format.
Run the Python script containing the ML model training and prediction logic.
Evaluate the model performance using appropriate metrics.
Make predictions on new data if required.

## Contributors
Rajendra Reang

## License
MIT License
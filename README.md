# Breast-Cancer-Classification-using-K-Nearest-Neighbors and Decision Tree Classifier

The dataset used in this project is the Breast Cancer Wisconsin (Diagnostic) dataset. This dataset is used for binary classification to distinguish between malignant and benign breast tumors based on various features extracted from breast mass images. Below is a brief summary of the dataset:

`Dataset Overview`<br>
Number of Features: 30 <br>
Number of Observations: 569 <br>

## Table of Contents

- [Project Overview](#project-overview)
- [Data Preparation](#data-preparation)
- [Feature Selection](#feature-selection)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Standardization](#standardization)
- [Decision Tree Classifier](#decision-tree-classifier)
- [Test Data Evaluation](#test-data-evaluation)
- [Plots and Results](#plots-and-results)
- [Conclusion](#conclusion)
- [Usage](#usage)

## Project Overview

This project aims to implement a K-Nearest Neighbors (KNN) classifier and a Decision Tree classifier to classify a dataset, utilizing feature selection and standardization techniques to optimize model performance. The dataset used is the Breast Cancer Wisconsin dataset.

## Data Preparation

The dataset is split into training and test sets. The training data is used for feature selection, hyperparameter tuning, and model training, while the test data is reserved for final evaluation.

## Feature Selection

A custom feature selection process is implemented using a Decision Tree Classifier to rank features based on importance. The KNN classifier is then trained incrementally with the most important features to observe the change in cross-validation accuracy and optimal k-values.

## Model Training and Evaluation

The KNN classifier is evaluated using 5-fold cross-validation to determine the mean cross-validation accuracy and the best k-value for each subset of features. This process is repeated until only one feature remains or the accuracy threshold condition is met.

## Standardization

The data is standardized to have a mean of zero and a standard deviation of one. The feature selection process is repeated on the standardized data, and results are compared with the non-standardized data.

## Decision Tree Classifier

A Decision Tree classifier is trained on the standardized data with hyperparameter tuning for `max_depth` and `min_samples_split` using cross-validation. The performance is compared with the KNN classifier.

## Test Data Evaluation

The best performing model (highest cross-validation accuracy) is applied to the test dataset to evaluate its performance. The best model had a test accuracy of 95%.

## Plots and Results

1. **Cross-Validation Accuracy vs. Number of Features**: This plot shows the mean cross-validation accuracy for each number of features, comparing both standardized and non-standardized data.
2. **Best k-value vs. Number of Features**: This plot displays the optimal k-values corresponding to each number of features for both standardized and non-standardized data.

## Conclusion

Standardization of the data significantly improves the cross-validation accuracy of the KNN classifier. The KNN classifier outperforms the Decision Tree classifier when using all features. The essential features for classification are determined through feature importance ranking.

## Usage

### Prerequisites

- Python 3.x
- Scikit-learn
- Numpy
- Matplotlib

### Running the Code

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/knn-feature-selection.git
    ```
2. Navigate to the project directory:
    ```bash
    cd knn-feature-selection
    ```
3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
4. Run the Jupyter Notebook or Python script to perform feature selection, model training, and evaluation:
    ```bash
    jupyter notebook knn_feature_selection.ipynb
    ```

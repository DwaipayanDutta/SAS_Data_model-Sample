# Logistic Regression with Synthetic Data: A SAS Data Model Example

This repository showcases the process of creating a logistic regression model in SAS using synthetic data generated by a Python script. The goal is to predict a binary outcome variable based on several predictor variables.

## Table of Contents

- [Overview](#overview) 📖
- [Logistic Regression](#logistic-regression) 📊
- [Information Value](#information-value) 📈
- [Confusion Matrix](#confusion-matrix) 📉
- [Getting Started](#getting-started) 🚀
- [Usage](#usage) 💻
- [License](#license) 📜

## Overview

Logistic regression is a popular statistical method for binary classification problems. This project guides you through generating synthetic data, building a logistic regression model in SAS, and evaluating its performance.

## Logistic Regression

Logistic regression is used to predict binary or categorical outcomes. It estimates the probability of an event occurring based on predictor variables. The model fits a logistic function to the data, resulting in a sigmoid curve that maps predicted values to probabilities between 0 and 1.

The logistic regression formula is:

$$
\log\left(\frac{p}{1-p}\right) = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_n x_n
$$

where:
- **p** is the probability of the event occurring
- **β₀** is the intercept
- **β₁, β₂, ..., βₙ** are the coefficients for predictor variables **x₁, x₂, ..., xₙ**

## Information Value

Information Value (IV) measures the predictive power of a variable in binary classification models. It quantifies the difference between the distributions of good and bad events for each predictor variable. A higher IV indicates better predictive capability.

The formula for IV is:

$$
IV = \sum_{i=1}^n (Good_i - Bad_i) \times \log\left(\frac{Good_i}{Bad_i}\right)
$$

where:
- **n** is the number of distinct values for the predictor variable
- **Goodᵢ** is the proportion of good events for the i-th value
- **Badᵢ** is the proportion of bad events for the i-th value

## Confusion Matrix

A confusion matrix provides a summary of a classification model’s performance. It displays the counts of true positives (TP), true negatives (TN), false positives (FP), and false negatives (FN) for a binary classification task.

The confusion matrix is structured as follows:

|                    | Predicted Positive | Predicted Negative |
|--------------------|--------------------|--------------------|
| **Actual Positive** | TP                 | FN                 |
| **Actual Negative** | FP                 | TN                 |

From this matrix, various performance metrics can be derived, including accuracy, precision, recall, and F1-score.

## Getting Started

To build the logistic regression model in SAS with synthetic data, follow these steps:

1. **Generate Synthetic Data**: <img src="https://cdn-icons-png.flaticon.com/512/3186/3186684.png" width="24" height="24" /> Run the Python script `data_generator.py` to create the synthetic dataset.
2. **Import Data into SAS**: <img src="https://cdn-icons-png.flaticon.com/512/2921/2921222.png" width="24" height="24" /> Load the generated data into SAS. 📊
3. **Perform Exploratory Data Analysis (EDA)**: <img src="https://cdn-icons-png.flaticon.com/512/181/181550.png" width="24" height="24" /> Analyze the data to understand relationships between predictors and the target variable. 🔍
4. **Split the Data**: <img src="https://cdn-icons-png.flaticon.com/512/2515/2515183.png" width="24" height="24" /> Divide the data into training and validation sets. 📈
5. **Train the Model**: <img src="https://cdn-icons-png.flaticon.com/512/1069/1069754.png" width="24" height="24" /> Use the training data to build the logistic regression model. 📉
6. **Evaluate Model Performance**: <img src="https://cdn-icons-png.flaticon.com/512/217/217964.png" width="24" height="24" /> Assess the model using metrics such as accuracy, AUC, and the confusion matrix. 🏆
7. **Fine-Tune the Model**: <img src="https://cdn-icons-png.flaticon.com/512/2169/2169821.png" width="24" height="24" /> Adjust the model parameters if needed and re-evaluate. ⚙️
8. **Deploy the Model**: <img src="https://cdn-icons-png.flaticon.com/512/1828/1828859.png" width="24" height="24" /> Apply the final model to make predictions on new data. 🚀

## Usage

To generate synthetic data and model, execute the following command in your terminal:

```bash
python Master_File.py

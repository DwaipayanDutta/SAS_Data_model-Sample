# SAS Data Model Sample: Logistic Regression with Synthetic Data

This repository demonstrates the creation of a logistic regression model in SAS using synthetic data generated by a Python script. The model aims to predict a binary outcome variable based on several predictor variables.

## Table of Contents

- [Overview](#overview)
- [Logistic Regression](#logistic-regression)
- [Information Value](#information-value)
- [Confusion Matrix](#confusion-matrix)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [License](#license)

## Overview

Logistic regression is a widely used statistical method for binary classification. This project provides a step-by-step guide to generating synthetic data, building a logistic regression model in SAS, and evaluating its performance.

## Logistic Regression

Logistic regression is a statistical model used for predicting binary or categorical outcomes. It estimates the probability of an event occurring based on one or more predictor variables. The model fits a logistic function to the data, which is a sigmoid curve that maps predicted values to probabilities between 0 and 1.

The logistic regression equation is:

$$\log\left(\frac{p}{1-p}\right) = \beta_0 + \beta_1x_1 + \beta_2x_2 + ... + \beta_nx_n$$

where:
- **p** is the probability of the event occurring
- **β₀** is the intercept term
- **β₁, β₂, ..., βₙ** are the coefficients for each predictor variable **x₁, x₂, ..., xₙ**

## Information Value

Information value (IV) is a measure of the predictive power of a variable in a binary classification model. It quantifies the difference between the distributions of good and bad events for each predictor variable. A higher IV indicates that a variable is more predictive of the target variable.

The formula for IV is:

$$IV = \sum_{i=1}^n (Good_i - Bad_i) \times \log\left(\frac{Good_i}{Bad_i}\right)$$

where:
- **n** is the number of distinct values for the predictor variable
- **Goodᵢ** is the proportion of good events for the i-th value
- **Badᵢ** is the proportion of bad events for the i-th value

## Confusion Matrix

A confusion matrix is a table that summarizes the performance of a classification model. It shows the number of true positives (TP), true negatives (TN), false positives (FP), and false negatives (FN) for a binary classification problem.

The confusion matrix looks like this:

|      | Predicted Positive | Predicted Negative |
|------|-------------------|-------------------|
| Actual Positive | TP | FN |
| Actual Negative | FP | TN |

From the confusion matrix, various performance metrics can be calculated, such as accuracy, precision, recall, and F1-score.

## Getting Started

To create the logistic regression model in SAS using synthetic data, follow these steps:

1. **Generate synthetic data** using the Python script `data_generator.py` provided in the repository. :snake:
2. **Import the generated data into SAS**. :bar_chart:
3. **Perform exploratory data analysis** to understand the relationships between the predictor variables and the target variable. :mag:
4. **Split the data into training and validation sets**. :chart_with_upwards_trend:
5. **Train the logistic regression model** using the training data. :chart_with_downwards_trend:
6. **Evaluate the model's performance** on the validation set using metrics such as accuracy, AUC, and the confusion matrix. :trophy:
7. **Fine-tune the model** if necessary and repeat steps 5-6. :wrench:
8. **Deploy the final model** for making predictions on new data. :rocket:

## Usage

To run the Python script for generating synthetic data, execute the following command:

```bash
python data_generator.py

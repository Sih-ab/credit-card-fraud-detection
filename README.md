# 💳 Credit Card Fraud Detection

A beginner-level Machine Learning project to detect fraudulent credit card transactions using Python and Logistic Regression.

## 📌 Project Overview

Credit card fraud detection is a classification problem where we try to identify whether a transaction is:

- `0` → Normal Transaction
- `1` → Fraud Transaction

In this project, I explored the dataset, checked missing values, analyzed class imbalance, trained a Logistic Regression model, and evaluated its performance.

## 📊 Dataset

Dataset: Credit Card Fraud Detection

Source: Kaggle

The dataset contains:

- 284,807 transactions
- 30 input features
- 1 target column (`Class`)

The dataset is highly imbalanced:

- Normal transactions: 284,315
- Fraud transactions: 492

## 🛠️ Technologies Used

- Python
- Pandas
- Matplotlib
- Scikit-learn
- Logistic Regression

## 🔍 Steps Performed

1. Loaded the dataset using Pandas
2. Explored the dataset
3. Checked missing values
4. Analyzed normal vs fraud transactions
5. Split the data into training and testing sets
6. Trained a Logistic Regression model
7. Evaluated the model using:
   - Accuracy
   - Precision
   - Recall
   - Confusion Matrix

## 🤖 Model

The model used in this project is:

**Logistic Regression**

Training data:

`227,845 rows`

Testing data:

`56,962 rows`

## 📈 Results

| Metric | Score |
|---|---:|
| Accuracy | 99.93% |
| Precision | 86.59% |
| Recall | 72.45% |

### Confusion Matrix

```text
[[56853    11]
 [   27    71]]
import pandas as pd

df = pd.read_csv("data/creditcard.csv")

print(df.head())

print("Shape:", df.shape)

print("\nColumns:")
print(df.columns)

print("\nData Info:")
print(df.info())

print("\nTransaction Class Counts:")
print(df["Class"].value_counts())

fraud_percentage = df["Class"].mean() * 100

print("Fraud Percentage:", fraud_percentage)

import matplotlib.pyplot as plt

class_counts = df["Class"].value_counts()

class_counts.plot(kind="bar")

plt.xlabel("Transaction Class")
plt.ylabel("Number of Transactions")
plt.title("Normal vs Fraud Transactions")

plt.show()

print("\nMissing Values:")
print(df.isnull().sum().sum())
from sklearn.model_selection import train_test_split

X = df.drop("Class", axis=1)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

print("Model training completed!")

from sklearn.metrics import accuracy_score, precision_score, recall_score

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_predictions(y_test, y_pred)

plt.title("Confusion Matrix - Fraud Detection")
plt.show()
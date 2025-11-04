#  Employee Attrition Prediction using Logistic Regression

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.DataFrame({
    'Age': [28, 35, 42, 25, 30, 38, 41, 27, 50, 31],
    'Salary': [40000, 55000, 60000, 35000, 42000, 58000, 62000, 37000, 70000, 45000],
    'YearsAtCompany': [2, 6, 8, 1, 3, 7, 9, 2, 10, 4],
    'JobSatisfaction': [3, 4, 2, 3, 3, 4, 2, 3, 1, 4],
    'WorkLifeBalance': [2, 3, 1, 3, 2, 3, 2, 2, 1, 3],
    'Overtime': [1, 0, 1, 0, 1, 0, 1, 1, 1, 0],  # 1 = Yes, 0 = No
    'Attrition': [1, 0, 1, 0, 0, 0, 1, 1, 1, 0]  # 1 = Left, 0 = Stayed
})

X = data[['Age', 'Salary', 'YearsAtCompany', 'JobSatisfaction', 'WorkLifeBalance', 'Overtime']]
y = data['Attrition']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(" Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print("\n Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\n Classification Report:\n", classification_report(y_test, y_pred))

sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

new_employee = pd.DataFrame({
    'Age': [35],
    'Salary': [2000],
    'YearsAtCompany': [2],
    'JobSatisfaction': [3],
    'WorkLifeBalance': [2],
    'Overtime': [1]
    
})

prediction = model.predict(new_employee)
print("\n Prediction for New Employee:", "Will Leave" if prediction[0] == 1 else "Will Stay")

# -------------------------------
# Loan Approval Prediction - Mini ML Project
# -------------------------------

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# 1. Load dataset
# Save the dataset as 'loan_data.csv' in the same folder
data = pd.read_csv("loan_data.csv")
print("First 5 rows:\n", data.head())

# 2. Preprocessing
data['LoanAmount'].fillna(data['LoanAmount'].median(), inplace=True)
data['Credit_History'].fillna(data['Credit_History'].median(), inplace=True)
data['Gender'].fillna(data['Gender'].mode()[0], inplace=True)
data['Married'].fillna(data['Married'].mode()[0], inplace=True)
data['Dependents'].fillna(data['Dependents'].mode()[0], inplace=True)
data['Self_Employed'].fillna(data['Self_Employed'].mode()[0], inplace=True)

# Encode categorical features
cols = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']
le = LabelEncoder()
for col in cols:
    data[col] = le.fit_transform(data[col])

# 3. Features and target
X = data[['Gender','Married','Education','Self_Employed','ApplicantIncome',
          'CoapplicantIncome','LoanAmount','Credit_History','Property_Area']]
y = data['Loan_Status'].map({'Y':1,'N':0})  # Convert target to 1/0

# 4. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 6. Predictions and evaluation
y_pred = model.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Predict for a new applicant
new_applicant = pd.DataFrame({
    'Gender':[1],
    'Married':[0],
    'Education':[1],
    'Self_Employed':[0],
    'ApplicantIncome':[5000],
    'CoapplicantIncome':[0],
    'LoanAmount':[200],
    'Credit_History':[1],
    'Property_Area':[2]
})


prediction = model.predict(new_applicant)
print("\nLoan Approval Prediction (1=Approved, 0=Not Approved):", prediction[0])
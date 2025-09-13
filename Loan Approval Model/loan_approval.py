import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = pd.read_csv("LoanApprovalPrediction.csv")
#print(data.head(5))

#DATA PREPROCESSING AND VISUALIZATION
#Loan_ID is unique and has no correlation with any other column
#It can be safely removed
data.drop(['Loan_ID'], axis=1, inplace=True)

#get the number of columns with object as datatype
#Find categorical columns as ML models can't handle text like data directly
obj = (data.dtypes == 'object')
object_cols = list(obj[obj].index)
plt.figure(figsize=(12, len(object_cols)*5))  # height depends on number of columns
index = 1
#print("Categorical Variables: ", len(list(obj[obj].index)))

#display information graphically
"""
for i, col in enumerate(object_cols, start=1):
    plt.subplot(len(object_cols), 1, i)  # 1 column, multiple rows
    y = data[col].value_counts()
    sns.barplot(x=list(y.index), y=y)
    plt.xticks(rotation=90)
    plt.title(col)

plt.tight_layout(pad=3.0, h_pad=5.0)  # more spacing between graphs
plt.show()
"""

#transforming categorical columns into numeric type
label_encoder = preprocessing.LabelEncoder()
obj = (data.dtypes == 'object')
for col in list(obj[obj].index):
    data[col] = label_encoder.fit_transform(data[col])

#check categorical columns
#display
obj = (data.dtypes == 'object')
print("Categorical Variables: ", len(list(obj[obj].index)))
plt.figure(figsize=(12,6))
sns.heatmap(data.corr(), cmap='BrBG',fmt='.2f',linewidths=2,annot=True)
"""
sns.catplot(x="Gender", y="Married",
            hue="Loan_Status",
            kind="bar",
            data=data
            )
"""
plt.show()

#find missing value in dataset
for col in data.columns:
    data[col] = data[col].fillna(data[col].mean())
print(data.isna().sum())

#MODEL DEVELOPMENT
#splitting dataset
X = data.drop(['Loan_Status'], axis = 1)
Y= data['Loan_Status']
#print(X.shape, Y.shape)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=1)
#print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)
#Scaling the data
scaler = preprocessing.StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#using Logistic Regression
#Binary Classification problem of whether the loan is sanctioned or not
#initialize the model
model = LogisticRegression(max_iter=1000)
#train the model
model.fit(X_train_scaled, Y_train)
#predictions with test data
Y_pred = model.predict(X_test_scaled)

#MODEL EVALUATION
print("Accuracy: ", accuracy_score(Y_test, Y_pred))
print("Confusion Matrix:\n",confusion_matrix(Y_test, Y_pred))
print("Classification Report:\n", classification_report(Y_test, Y_pred))

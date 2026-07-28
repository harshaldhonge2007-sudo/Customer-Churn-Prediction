import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

Lencoder = LabelEncoder()
df_new = df.copy()

cols = ["gender", "Partner", "Dependents","PhoneService","PaperlessBilling"]
df_new.drop(['customerID'], axis=1, inplace=True)
df_new["TotalCharges"] = pd.to_numeric(df_new["TotalCharges"], errors='coerce')

onehot_cols = [
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaymentMethod"
]
df_new = pd.get_dummies(df_new,columns=onehot_cols)

df_new[cols] = df_new[cols].apply(LabelEncoder().fit_transform)

df_new.fillna(0, inplace=True)

model = LogisticRegression()

df_new["Churn"] = Lencoder.fit_transform(df_new["Churn"])

X = df_new.drop("Churn", axis=1)
y = df_new["Churn"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = LogisticRegression(max_iter=5000)

model.fit(X_train,y_train)
y_pred = model.predict(X_test)

print("Accuracy :", accuracy_score(y_test, y_pred))

print("\nClassification Report\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix\n")
print(confusion_matrix(y_test, y_pred))


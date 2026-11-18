import os
import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

data = pd.read_csv("dataset/students.csv")
data.columns = data.columns.str.strip()
data = data.dropna(subset=["result"])

X = data.drop(columns=["result"]).select_dtypes(include="number")
X = X.fillna(X.median())
y = data["result"].astype(str).str.strip()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc * 100:.2f}%")
print(classification_report(y_test, y_pred))

os.makedirs("models", exist_ok=True)
version = len(os.listdir("models")) + 1
path = f"models/v{version}.pkl"
with open(path, "wb") as f:
    pickle.dump({"model": model, "features": list(X.columns), "target": "result",
                 "dataset": "students_large.csv", "accuracy": acc}, f)
print("Saved:", path)
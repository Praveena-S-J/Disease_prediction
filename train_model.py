import pandas as pd
import json
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
accuracy_score,
precision_score,
recall_score,
f1_score,
classification_report
)

# Load dataset

df = pd.read_csv("Blood_samples_dataset_balanced.csv")

# Target column

TARGET_COLUMN = "Disease"

# Features and target

X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]

# Train-test split

X_train, X_test, y_train, y_test = train_test_split(
X,
y,
test_size=0.2,
random_state=42,
stratify=y
)

# Train model

model = RandomForestClassifier(
n_estimators=100,
random_state=42
)

model.fit(X_train, y_train)

# Predictions

y_pred = model.predict(X_test)

# Classification Report

report = classification_report(
y_test,
y_pred,
output_dict=True,
zero_division=0
)

# Metrics

metrics = {
"accuracy": float(accuracy_score(y_test, y_pred)),
"precision": float(
precision_score(y_test, y_pred, average="weighted", zero_division=0)
),
"recall": float(
recall_score(y_test, y_pred, average="weighted", zero_division=0)
),
"f1_score": float(
f1_score(y_test, y_pred, average="weighted", zero_division=0)
),
"n_train": int(len(X_train)),
"n_test": int(len(X_test)),
"classes": sorted([str(c) for c in y.unique()]),
"classification_report": report
}

print("classification_report" in metrics)
print(metrics.keys())

# Save model

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save metrics

with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("Model saved as model.pkl")
print("Metrics saved as metrics.json")
print(metrics)

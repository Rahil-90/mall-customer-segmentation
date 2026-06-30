import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("Mall_Customers.csv")

# Create target automatically
df["Segment"] = (df["Spending_Score"] >= 50).astype(int)

# Encode Gender
df["Genre"] = df["Genre"].map({
    "Male": 1,
    "Female": 0
})

# Features
X = df[
    [
        "Genre",
        "Age",
        "Annual_Income_(k$)",
        "Spending_Score"
    ]
]

# Target
y = df["Segment"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, preds))

# Save model
joblib.dump(model, "model.pkl")

print("Model saved successfully!")
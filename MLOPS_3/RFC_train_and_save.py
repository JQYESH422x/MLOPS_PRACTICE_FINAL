from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

# 1. Data & train
iris = load_iris(as_frame=True)
X_train, X_val, y_train, y_val = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 2. Evaluate (optional)
acc = accuracy_score(y_val, model.predict(X_val))
print(f"Validation accuracy: {acc:.3f}")

# 3. Save
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/iris_rf.pkl")
print("Model saved to models/iris_rf.pkl")

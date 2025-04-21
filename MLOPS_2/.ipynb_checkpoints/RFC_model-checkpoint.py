import os
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# 1. Data Preparation
iris = load_iris(as_frame=True)
X = iris.data
y = iris.target
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2. Define hyperparameter grid for max_depth
depths = [2, 4, 6]  # None means unlimited depth

# 3. Training, tuning & versioning
os.makedirs("models", exist_ok=True)
results = []
version = 1

for depth in depths:
    # 3a. Train
    model = RandomForestClassifier(
        max_depth=depth,
        n_estimators=100,        # fixed number of trees
        random_state=42
    )
    model.fit(X_train, y_train)

    # 3b. Evaluate
    preds = model.predict(X_val)
    acc = accuracy_score(y_val, preds)

    # 3c. Record & save
    results.append({
        "version": version,
        "max_depth": str(depth),
        "accuracy": acc
    })
    filename = f"models/rf_model_v{version}.pkl"
    joblib.dump(model, filename)
    print(f"Saved {filename} → acc={acc:.3f}")
    version += 1

# 4. Save all results
pd.DataFrame(results).to_csv("results_rf.csv", index=False)
print("Done. Results written to results_rf.csv")

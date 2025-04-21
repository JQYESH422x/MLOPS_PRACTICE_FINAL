import joblib
from sklearn.datasets import load_iris

# 1. Load saved model
model = joblib.load("models/iris_rf.pkl")

# 2. Load Iris dataset & prepare a sample
iris = load_iris()
sample = iris.data[0].reshape(1, -1)
actual_class = iris.target[0]
class_names = iris.target_names

# 3. Predict
pred = model.predict(sample)[0]

# 4. Display actual vs predicted
print(f"Input Features : {iris.feature_names}")
print(f"Sample Values  : {iris.data[0]}")
print(f"Actual Class   : {actual_class} ({class_names[actual_class]})")
print(f"Predicted Class: {pred} ({class_names[pred]})")

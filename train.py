import joblib
import numpy as np
from sklearn.linear_model import LinearRegression

X = np.array([[1], [2], [3], [4], [5], [6]])
y = np.array([3.5, 5.0, 6.5, 8.0, 10.0, 12.5])

model = LinearRegression().fit(X, y)
joblib.dump(model, "model.pkl")
print("Model trained and saved successfully!")

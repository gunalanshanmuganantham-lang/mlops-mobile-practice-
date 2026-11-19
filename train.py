import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Dataset
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8]])
y = np.array([3.5, 5.0, 6.5, 8.0, 10.0, 12.5, 14.5, 17.0])

# Candidate Models
models = {
    "Linear_Regression": LinearRegression(),
    "Decision_Tree": DecisionTreeRegressor(max_depth=3, random_state=42),
    "Random_Forest": RandomForestRegressor(n_estimators=50, random_state=42)
}

best_model = None
best_score = -float("inf")
best_name = ""

for name, model in models.items():
    model.fit(X, y)
    score = r2_score(y, model.predict(X))
    print(f"Model: {name} | R2 Score: {round(score, 4)}")
    
    if score > best_score:
        best_score = score
        best_model = model
        best_name = name

joblib.dump(best_model, "model.pkl")
print(f"🏆 Winner: {best_name} (R2: {round(best_score, 4)}) saved as model.pkl!")

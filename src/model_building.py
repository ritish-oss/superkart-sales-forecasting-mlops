import os, json, joblib, numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

os.makedirs("models", exist_ok=True)
train = pd.read_csv("artifacts/train.csv")
test = pd.read_csv("artifacts/test.csv")
X_tr, y_tr = train.drop(columns=["Product_Store_Sales_Total"]), train["Product_Store_Sales_Total"]
X_te, y_te = test.drop(columns=["Product_Store_Sales_Total"]), test["Product_Store_Sales_Total"]

num_cols = X_tr.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = X_tr.select_dtypes(include=['object']).columns.tolist()

prep = ColumnTransformer([('num', StandardScaler(), num_cols), ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)])
pipe = Pipeline([('prep', prep), ('model', RandomForestRegressor(random_state=42))])
grid = GridSearchCV(pipe, {'model__n_estimators': [100], 'model__max_depth': [8]}, cv=3, scoring='neg_root_mean_squared_error', n_jobs=-1)
grid.fit(X_tr, y_tr)
preds = grid.predict(X_te)
metrics = {"test_rmse": float(np.sqrt(mean_squared_error(y_te, preds))), "test_mae": float(mean_absolute_error(y_te, preds)), "test_r2": float(r2_score(y_te, preds))}
joblib.dump(grid.best_estimator_, "models/best_model.pkl")
with open("models/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)
print(f"[MODEL BUILDING SUCCESS] Metrics: {metrics}")

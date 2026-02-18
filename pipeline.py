import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor

np.random.seed(42)

trn = pd.read_csv('data/CW1_train.csv')
X_tst = pd.read_csv('data/CW1_test.csv')

#cleaning the outliers and incorrect values we found
trn = trn[(trn['x'] != 0) & (trn['y'] != 0) & (trn['z'] != 0)]
trn = trn[trn['y'] < 20]


print(f"Training rows after cleaning: {len(trn)} (supposed to drop 5)")


categorical_cols = ['cut', 'color', 'clarity']
trn = pd.get_dummies(trn, columns=categorical_cols, drop_first=True)
X_tst = pd.get_dummies(X_tst, columns=categorical_cols, drop_first=True)

# (in case a category appears in one dataset but not the other)
X_tst = X_tst.reindex(columns=trn.drop(columns=['outcome']).columns, fill_value=0)

X = trn.drop(columns=['outcome'])
y = trn['outcome']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Train: {len(X_train)}, Validation: {len(X_val)}")
print(f"Features: {X_train.shape[1]}")

final_model = XGBRegressor(
    n_estimators=1000,
    max_depth=3,
    learning_rate=0.01,
    subsample=0.8,
    random_state=42
)
final_model.fit(X, y)

# Predict on test set
yhat = final_model.predict(X_tst)

# Save submission
out = pd.DataFrame({'yhat': yhat})
out.to_csv('CW1_submission_23094651.csv', index=False)
print(f"Submission saved: {len(yhat)} predictions")

#model comparison (commented out now)
'''
models = {
    'Linear Regression': LinearRegression(),
    'Ridge':             Ridge(alpha=1.0),
    'Lasso':             Lasso(alpha=0.1),
    'Random Forest':     RandomForestRegressor(n_estimators=200, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, random_state=42),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    yhat = model.predict(X_val)
    r2 = r2_score(y_val, yhat)
    results[name] = r2
    print(f"{name:25s} - Validation R²: {r2:.4f}")

print(f"\nBest: {max(results, key=results.get)}")


param_grid = {
    'n_estimators': [200, 500, 1000],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 1.0],
}

gb = GradientBoostingRegressor(random_state=42)

# Option A: fit on X_train only, validate on X_val
grid_a = GridSearchCV(gb, param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=1)
grid_a.fit(X_train, y_train)
print(f"Option A - Best CV R²: {grid_a.best_score_:.4f}")
print(f"Option A - Val R²: {r2_score(y_val, grid_a.best_estimator_.predict(X_val)):.4f}")

# Option B: fit on full X, rely on CV score
grid_b = GridSearchCV(gb, param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=1)
grid_b.fit(X, y)
print(f"Option B - Best CV R²: {grid_b.best_score_:.4f}")


print(f"Best params A: {grid_a.best_params_}")
print(f"Best params B: {grid_b.best_params_}")


#final experiment
xgb_params = {
    'n_estimators': [200, 500, 1000],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 1.0],
    'tree_method': 'hist',
    'device': 'cpu',
}

xgb = XGBRegressor(random_state=42, tree_method='hist', device='cpu')
xgb_grid = GridSearchCV(
    xgb,
    {k: v for k, v in xgb_params.items() if isinstance(v, list)},
    cv=5, scoring='r2', n_jobs=-1, verbose=1
)
xgb_grid.fit(X_train, y_train)

print(f"XGBoost - Best CV R²: {xgb_grid.best_score_:.4f}")
print(f"XGBoost - Val R²: {r2_score(y_val, xgb_grid.best_estimator_.predict(X_val)):.4f}")
print(f"XGBoost - Best params: {xgb_grid.best_params_}")
'''
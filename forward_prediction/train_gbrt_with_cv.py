from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, ParameterGrid
from sklearn.multioutput import MultiOutputRegressor


# -----------------------------
# 1) Load data
# -----------------------------
DATA_DIR = Path("forward_prediction/split")
RESULTS_FILE = Path("results.txt")

X_train = pd.read_csv(DATA_DIR / "train_features.csv")
X_test = pd.read_csv(DATA_DIR / "test_features.csv")
y_train = pd.read_csv(DATA_DIR / "train_labels.csv")
y_test = pd.read_csv(DATA_DIR / "test_labels.csv")

# Remove row_id if present (it is an index, not a real feature/target).
for df in (X_train, X_test, y_train, y_test):
    if "row_id" in df.columns:
        df.drop(columns=["row_id"], inplace=True)

print("Data loaded.")
print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_test : {X_test.shape}, y_test : {y_test.shape}")


# -----------------------------
# 2) Build model + grid search
# -----------------------------
base_model = GradientBoostingRegressor(random_state=42)
model = MultiOutputRegressor(base_model)

param_grid = {
    "estimator__n_estimators": [180, 200, 220], #number of boosting stages (number of trees)
    "estimator__learning_rate": [0.015, 0.025, 0.035], #step size of each boosting stage
    "estimator__max_depth": [4, 5, 6], #maximum depth of each regression tree 
    "estimator__subsample": [0.5, 0.6], #fraction of samples used for fitting each base learner
    "estimator__min_samples_leaf": [5, 10, 12], #minimum number of samples required to be at a leaf node (= minimum number of samples in each child node to do a split)
}

cv = KFold(n_splits=5, shuffle=True, random_state=42)
num_candidates = len(ParameterGrid(param_grid))
total_fits = num_candidates * cv.get_n_splits()

print("\nStarting hyperparameter search...")
print(f"Candidates: {num_candidates}")
print(f"CV folds  : {cv.get_n_splits()}")
print(f"Total fits: {total_fits}")

# GridSearchCV tries all combinations of the hyperparameters above
# and selects the one with the best cross-validation score (lowest MAE in this case).
search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring="neg_mean_absolute_error",
    cv=cv,
    n_jobs=-1,
    verbose=1,
)

start_time = time.time()
search.fit(X_train, y_train)
elapsed = time.time() - start_time

print("\nBest hyperparameters:")
print(search.best_params_)
print(f"Best CV MAE: {-search.best_score_:.6f}")
print(f"Training time: {elapsed:.1f} seconds")


# -----------------------------
# 3) Evaluate on test set
# -----------------------------
best_model = search.best_estimator_
y_pred = best_model.predict(X_test)

y_true = y_test.to_numpy()

mae_overall = mean_absolute_error(y_true, y_pred)
rmse_overall = np.sqrt(mean_squared_error(y_true, y_pred))
r2_overall = r2_score(y_true, y_pred, multioutput="uniform_average")

print("\nTest set (overall):")
print(f"MAE  : {mae_overall:.6f}")
print(f"RMSE : {rmse_overall:.6f}")
print(f"R2   : {r2_overall:.6f}")


# -----------------------------
# 4) Challenge benchmark score
# -----------------------------
TARGET_WEIGHTS = {
    "P80": 0.30,
    "R95": 0.20,
    "fines_frac": 0.15,
    "oversize_frac": 0.15,
    "R50_fines": 0.10,
    "R50_oversize": 0.10,
}


# sMAPE (symmetric mean absolute percentage error) for fractional targets,
def smape(y_true_1d: np.ndarray, y_pred_1d: np.ndarray, eps: float = 1e-12) -> float:
    denominator = np.abs(y_true_1d) + np.abs(y_pred_1d) + eps
    return float(np.mean(2.0 * np.abs(y_pred_1d - y_true_1d) / denominator))


challenge_errors: dict[str, float] = {}

for target in TARGET_WEIGHTS:
    y_t = y_test[target].to_numpy()
    y_p = y_pred[:, y_test.columns.get_loc(target)]

    if target in {"fines_frac", "oversize_frac"}:
        challenge_errors[target] = smape(y_t, y_p)
    else:
        challenge_errors[target] = float(mean_absolute_error(y_t, y_p))

weighted_error = sum(TARGET_WEIGHTS[t] * challenge_errors[t] for t in TARGET_WEIGHTS)
challenge_score = 100.0 / (1.0 + weighted_error)

print("\nChallenge benchmark:")
for target in TARGET_WEIGHTS:
    metric_name = "sMAPE" if target in {"fines_frac", "oversize_frac"} else "MAE"
    print(
        f"{target:>15} | {metric_name}={challenge_errors[target]:.6f}  "
        f"weight={TARGET_WEIGHTS[target]:.2f}"
    )
print(f"Weighted error: {weighted_error:.6f}")
print(f"Challenge score: {challenge_score:.6f}")

print("\nTest set (per target):")
for i, target_name in enumerate(y_test.columns):
    y_t = y_true[:, i]
    y_p = y_pred[:, i]
    mae = mean_absolute_error(y_t, y_p)
    rmse = np.sqrt(mean_squared_error(y_t, y_p))
    r2 = r2_score(y_t, y_p)
    print(f"{target_name:>15} | MAE={mae:.6f}  RMSE={rmse:.6f}  R2={r2:.6f}")


# -----------------------------
# 5) Append run summary to results.txt
# -----------------------------
result_lines = [
    "",
    "=" * 80,
    f"Script: {Path(__file__).name}",
    "Model specification:",
    "  - MultiOutputRegressor(GradientBoostingRegressor(random_state=42))",
    f"  - Best hyperparameters: {search.best_params_}",
    "  - CV: KFold(n_splits=5, shuffle=True, random_state=42)",
    "  - Scoring: neg_mean_absolute_error",
    "",
    "Overall test metrics:",
    f"  MAE={mae_overall:.6f}",
    f"  RMSE={rmse_overall:.6f}",
    f"  R2={r2_overall:.6f}",
    "",
    "Challenge benchmark:",
]

for target in TARGET_WEIGHTS:
    metric_name = "sMAPE" if target in {"fines_frac", "oversize_frac"} else "MAE"
    result_lines.append(
        f"  {target}: {metric_name}={challenge_errors[target]:.6f}, weight={TARGET_WEIGHTS[target]:.2f}"
    )

result_lines.extend(
    [
        f"  Weighted error={weighted_error:.6f}",
        f"  Challenge score={challenge_score:.6f}",
        "",
        "Per-target test metrics:",
    ]
)

for i, target_name in enumerate(y_test.columns):
    y_t = y_true[:, i]
    y_p = y_pred[:, i]
    mae = mean_absolute_error(y_t, y_p)
    rmse = np.sqrt(mean_squared_error(y_t, y_p))
    r2 = r2_score(y_t, y_p)
    result_lines.append(
        f"  {target_name}: MAE={mae:.6f}, RMSE={rmse:.6f}, R2={r2:.6f}"
    )

with RESULTS_FILE.open("a", encoding="utf-8") as f:
    f.write("\n".join(result_lines) + "\n")

print(f"\nRun summary appended to {RESULTS_FILE}")
